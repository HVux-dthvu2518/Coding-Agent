//! A small blocking HTTP/1.1 client for talking to a local Ollama daemon.
//!
//! Why hand-rolled: crates.io is not reachable from the deployment target, and
//! the only requirement is `POST /api/chat` against `localhost`. The client
//! therefore implements exactly what that needs and nothing else:
//!
//! * absolute `http://host[:port][/path]` URLs,
//! * `GET` and `POST` with a JSON body,
//! * `Content-Length`, `Transfer-Encoding: chunked` and read-until-close bodies,
//! * line-oriented reads for NDJSON streaming,
//! * connect and read-inactivity timeouts.
//!
//! Not implemented on purpose: TLS, redirects, cookies, HTTP/2, proxies, gzip.
//! Those become free once the crate can depend on `reqwest` (see README).

use crate::error::HttpError;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// Default connect timeout.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Default per-read inactivity timeout. Local models can pause between tokens,
/// so this is deliberately generous.
pub const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(300);

/// A parsed `http://` URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Url {
    host: String,
    port: u16,
    path: String,
}

impl Url {
    /// Parses `http://host[:port][/path]`.
    pub fn parse(raw: &str) -> Result<Url, HttpError> {
        let rest = raw
            .strip_prefix("http://")
            .ok_or_else(|| HttpError::BadUrl(format!("`{raw}` must start with `http://`")))?;
        if rest.is_empty() {
            return Err(HttpError::BadUrl("missing host".into()));
        }
        let (authority, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        };
        if authority.is_empty() {
            return Err(HttpError::BadUrl("missing host".into()));
        }
        if authority.contains('@') {
            return Err(HttpError::BadUrl(
                "credentials in URLs are not supported".into(),
            ));
        }
        let (host, port) = match authority.rsplit_once(':') {
            Some((h, p)) => {
                let port = p
                    .parse::<u16>()
                    .map_err(|_| HttpError::BadUrl(format!("invalid port `{p}`")))?;
                (h.to_string(), port)
            }
            None => (authority.to_string(), 80),
        };
        if host.is_empty() {
            return Err(HttpError::BadUrl("missing host".into()));
        }
        Ok(Url {
            host,
            port,
            path: path.to_string(),
        })
    }

    pub fn host(&self) -> &str {
        &self.host
    }

    /// Path (and query, if any) sent on the request line.
    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    /// Path plus a query string joined onto the URL's existing path.
    pub fn join(&self, endpoint: &str) -> Url {
        let mut path = self.path.trim_end_matches('/').to_string();
        if !endpoint.starts_with('/') {
            path.push('/');
        }
        path.push_str(endpoint);
        Url {
            host: self.host.clone(),
            port: self.port,
            path,
        }
    }

    /// Host header value, omitting the default port.
    fn host_header(&self) -> String {
        if self.port == 80 {
            self.host.clone()
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }
}

/// A streaming HTTP response. The body is readable line by line, which is the
/// natural shape of Ollama's newline-delimited JSON.
pub struct Response {
    status: u16,
    body: Body<Prefixed<TcpStream>>,
}

impl Response {
    pub fn status(&self) -> u16 {
        self.status
    }

    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// Reads the next line without its trailing CR/LF. `Ok(None)` at EOF.
    ///
    /// Implemented byte-wise over [`Response::read`] so that all three body
    /// framings (length, chunked, until-EOF) support line-oriented streaming.
    pub fn read_line(&mut self) -> Result<Option<String>, HttpError> {
        let mut bytes = Vec::new();
        let mut byte = [0u8; 1];
        loop {
            match self.read(&mut byte)? {
                0 => break,
                _ => {
                    if byte[0] == b'\n' {
                        break;
                    }
                    bytes.push(byte[0]);
                }
            }
        }
        if bytes.is_empty() && byte[0] != b'\n' {
            return Ok(None);
        }
        while matches!(bytes.last(), Some(b'\r')) {
            bytes.pop();
        }
        Ok(Some(String::from_utf8_lossy(&bytes).into_owned()))
    }

    /// Reads up to `buf.len()` raw body bytes. `Ok(0)` means the body ended.
    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize, HttpError> {
        self.body
            .read(buf)
            .map_err(|e| HttpError::Read(e.to_string()))
    }

    /// Reads the whole body as UTF-8 (lossy for invalid bytes).
    pub fn read_to_string(&mut self) -> Result<String, HttpError> {
        let mut buf = Vec::new();
        self.body
            .read_to_end(&mut buf)
            .map_err(|e| HttpError::Read(e.to_string()))?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }
}

/// Sends a `GET` and returns the response once headers are parsed.
pub fn get(url: &Url) -> Result<Response, HttpError> {
    request(url, "GET", None)
}

/// Sends a `POST` with a JSON body.
pub fn post_json(url: &Url, body: &[u8]) -> Result<Response, HttpError> {
    request(url, "POST", Some(body))
}

fn request(url: &Url, method: &str, body: Option<&[u8]>) -> Result<Response, HttpError> {
    let mut head = format!(
        "{method} {} HTTP/1.1\r\nHost: {}\r\nAccept: application/json\r\nConnection: close\r\nUser-Agent: qwen-agent/{}\r\n",
        url.path,
        url.host_header(),
        env!("CARGO_PKG_VERSION"),
    );
    if let Some(body) = body {
        head.push_str("Content-Type: application/json\r\n");
        head.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    head.push_str("\r\n");

    let stream = connect(url)?;

    let mut writer = stream
        .try_clone()
        .map_err(|e| HttpError::Connect(format!("cannot clone socket: {e}")))?;
    writer
        .write_all(head.as_bytes())
        .and_then(|()| match body {
            Some(b) => writer.write_all(b),
            None => Ok(()),
        })
        .and_then(|()| writer.flush())
        .map_err(|e| HttpError::Write(e.to_string()))?;
    drop(writer);

    // Read the head through a BufReader: it will over-read into the body, so
    // its leftover buffer is replayed by `Prefixed` before the socket.
    let mut reader = BufReader::new(
        stream
            .try_clone()
            .map_err(|e| HttpError::Connect(format!("cannot clone socket: {e}")))?,
    );
    let (status, headers) = read_head(&mut reader)?;
    let leftover = reader.buffer().to_vec();

    let framing = if headers
        .iter()
        .any(|(k, v)| k == "transfer-encoding" && v.to_ascii_lowercase().contains("chunked"))
    {
        Framing::Chunked
    } else if let Some((_, v)) = headers.iter().find(|(k, _)| k == "content-length") {
        let n = v
            .trim()
            .parse::<u64>()
            .map_err(|_| HttpError::Protocol(format!("invalid Content-Length `{v}`")))?;
        Framing::Length(n)
    } else {
        Framing::UntilEof
    };

    let transport = Prefixed {
        prefix: leftover,
        inner: stream,
    };
    Ok(Response {
        status,
        body: Body::new(transport, framing),
    })
}

fn connect(url: &Url) -> Result<TcpStream, HttpError> {
    let addrs = (url.host(), url.port())
        .to_socket_addrs()
        .map_err(|e| HttpError::Connect(format!("cannot resolve {}: {e}", url.host())))?;
    let mut last = String::from("no addresses resolved");
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, DEFAULT_CONNECT_TIMEOUT) {
            Ok(stream) => {
                stream
                    .set_read_timeout(Some(DEFAULT_READ_TIMEOUT))
                    .map_err(|e| HttpError::Connect(e.to_string()))?;
                stream
                    .set_nodelay(true)
                    .map_err(|e| HttpError::Connect(e.to_string()))?;
                return Ok(stream);
            }
            Err(e) => last = format!("{addr}: {e}"),
        }
    }
    Err(HttpError::Connect(last))
}

type Headers = Vec<(String, String)>;

fn read_head<R: BufRead>(reader: &mut R) -> Result<(u16, Headers), HttpError> {
    let status_line = read_crlf_line(reader)?
        .ok_or_else(|| HttpError::Protocol("connection closed before status line".into()))?;
    let mut parts = status_line.split_whitespace();
    let version = parts.next().unwrap_or_default();
    if !version.starts_with("HTTP/1.") {
        return Err(HttpError::Protocol(format!(
            "expected an HTTP/1.x status line, got `{status_line}`"
        )));
    }
    let status = parts
        .next()
        .and_then(|s| s.parse::<u16>().ok())
        .ok_or_else(|| HttpError::Protocol(format!("bad status code in `{status_line}`")))?;

    let mut headers = Headers::new();
    loop {
        let line = read_crlf_line(reader)?
            .ok_or_else(|| HttpError::Protocol("connection closed inside headers".into()))?;
        if line.is_empty() {
            return Ok((status, headers));
        }
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
        }
    }
}

fn read_crlf_line<R: BufRead>(reader: &mut R) -> Result<Option<String>, HttpError> {
    let mut buf = Vec::new();
    let n = reader
        .read_until(b'\n', &mut buf)
        .map_err(|e| HttpError::Read(e.to_string()))?;
    if n == 0 {
        return Ok(None);
    }
    while matches!(buf.last(), Some(b'\n') | Some(b'\r')) {
        buf.pop();
    }
    Ok(Some(String::from_utf8_lossy(&buf).into_owned()))
}

/// How the response body is delimited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Framing {
    /// `Content-Length: n`
    Length(u64),
    /// `Transfer-Encoding: chunked`
    Chunked,
    /// Body lasts until the peer closes the connection.
    UntilEof,
}

/// A reader that yields buffered bytes before continuing with the inner reader.
///
/// Needed because the HTTP head is parsed with a `BufReader`, which legitimately
/// reads past the blank line into the body. Dropping that buffer would lose the
/// first bytes of every chunked stream — the classic "first record is truncated"
/// bug, which no amount of NDJSON buffering can repair.
struct Prefixed<R: Read> {
    prefix: Vec<u8>,
    inner: R,
}

impl<R: Read> Read for Prefixed<R> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if !self.prefix.is_empty() {
            let n = self.prefix.len().min(out.len());
            out[..n].copy_from_slice(&self.prefix[..n]);
            self.prefix.drain(..n);
            return Ok(n);
        }
        self.inner.read(out)
    }
}

/// Decodes the three supported body framings into a plain byte stream.
struct Body<R: Read> {
    reader: R,
    framing: Framing,
    /// Bytes of the current chunk not yet handed to the caller.
    pending: Vec<u8>,
    pending_pos: usize,
    finished: bool,
}

impl<R: Read> Body<R> {
    fn new(reader: R, framing: Framing) -> Self {
        Self {
            reader,
            framing,
            pending: Vec::new(),
            pending_pos: 0,
            finished: false,
        }
    }
}

impl<R: Read> Read for Body<R> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if out.is_empty() || self.finished {
            return Ok(0);
        }
        if self.pending_pos < self.pending.len() {
            let n = (self.pending.len() - self.pending_pos).min(out.len());
            out[..n].copy_from_slice(&self.pending[self.pending_pos..self.pending_pos + n]);
            self.pending_pos += n;
            if self.pending_pos == self.pending.len() {
                self.pending.clear();
                self.pending_pos = 0;
            }
            return Ok(n);
        }
        match self.framing {
            Framing::UntilEof => self.reader.read(out),
            Framing::Length(remaining) => {
                if remaining == 0 {
                    self.finished = true;
                    return Ok(0);
                }
                let want = out.len().min(remaining as usize);
                let n = self.reader.read(&mut out[..want])?;
                if n == 0 {
                    self.finished = true;
                    return Ok(0);
                }
                self.framing = Framing::Length(remaining - n as u64);
                Ok(n)
            }
            Framing::Chunked => self.read_chunked(out),
        }
    }
}

impl<R: Read> Body<R> {
    fn read_chunked(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        loop {
            // At a chunk boundary: read "size[;ext]CRLF".
            let line = match read_line_lf(&mut self.reader)? {
                Some(line) => line,
                None => {
                    self.finished = true;
                    return Ok(0);
                }
            };
            let line = line.trim();
            if line.is_empty() {
                continue; // tolerate a stray blank line between chunks
            }
            let hex = line.split(';').next().unwrap_or(line).trim();
            let size = usize::from_str_radix(hex, 16).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid chunk size `{hex}`"),
                )
            })?;
            if size == 0 {
                // Consume optional trailers up to the terminating blank line.
                while let Some(t) = read_line_lf(&mut self.reader)? {
                    if t.is_empty() {
                        break;
                    }
                }
                self.finished = true;
                return Ok(0);
            }

            // Read the whole chunk so CRLF framing stays aligned, then hand
            // back as much as `out` can take (the rest waits in `pending`).
            let mut chunk = vec![0u8; size];
            self.reader.read_exact(&mut chunk)?;
            skip_crlf(&mut self.reader)?;

            let n = size.min(out.len());
            out[..n].copy_from_slice(&chunk[..n]);
            if n < size {
                self.pending = chunk;
                self.pending_pos = n;
            }
            return Ok(n);
        }
    }
}

/// Reads one LF-terminated line from a plain `Read` (chunk headers tiny).
fn read_line_lf<R: Read>(reader: &mut R) -> std::io::Result<Option<String>> {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        match reader.read(&mut byte) {
            Ok(0) => {
                if buf.is_empty() {
                    return Ok(None);
                }
                break;
            }
            Ok(_) => {
                if byte[0] == b'\n' {
                    break;
                }
                buf.push(byte[0]);
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    while matches!(buf.last(), Some(b'\r')) {
        buf.pop();
    }
    Ok(Some(String::from_utf8_lossy(&buf).into_owned()))
}

fn skip_crlf<R: Read>(reader: &mut R) -> std::io::Result<()> {
    let mut buf = [0u8; 2];
    let n = reader.read(&mut buf)?;
    if n == 0 {
        return Ok(());
    }
    if buf[0] == b'\r' || buf[0] == b'\n' {
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "missing CRLF after chunk",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parses_urls() {
        let u = Url::parse("http://localhost:11434").unwrap();
        assert_eq!((u.host(), u.port()), ("localhost", 11434));
        assert_eq!(u.join("/api/chat").path, "/api/chat");

        let u = Url::parse("http://127.0.0.1").unwrap();
        assert_eq!(u.port(), 80);
        assert_eq!(u.join("api/tags").path, "/api/tags");

        // Base URLs with a path prefix keep it.
        let u = Url::parse("http://box:8080/ollama/").unwrap();
        assert_eq!(u.join("/api/chat").path, "/ollama/api/chat");
    }

    #[test]
    fn rejects_bad_urls() {
        assert!(matches!(Url::parse("https://x"), Err(HttpError::BadUrl(_))));
        assert!(matches!(
            Url::parse("localhost:11434"),
            Err(HttpError::BadUrl(_))
        ));
        assert!(matches!(
            Url::parse("http://host:notaport"),
            Err(HttpError::BadUrl(_))
        ));
        assert!(matches!(
            Url::parse("http://user:pw@host"),
            Err(HttpError::BadUrl(_))
        ));
        assert!(matches!(Url::parse("http://"), Err(HttpError::BadUrl(_))));
    }

    fn body_of(raw: &str) -> Body<Cursor<Vec<u8>>> {
        // Frame selection happens in `request`; here we test the reader itself.
        Body::new(Cursor::new(raw.as_bytes().to_vec()), Framing::Chunked)
    }

    #[test]
    fn prefixed_replays_buffered_body_bytes() {
        let mut reader = Prefixed {
            prefix: b"{\"content\":".to_vec(),
            inner: Cursor::new(b"\"hi\"}".to_vec()),
        };
        let mut out = String::new();
        reader.read_to_string(&mut out).unwrap();
        assert_eq!(out, "{\"content\":\"hi\"}");

        // Byte-at-a-time reads must drain the prefix exactly.
        let mut reader = Prefixed {
            prefix: b"ab".to_vec(),
            inner: Cursor::new(b"cd".to_vec()),
        };
        let mut one = [0u8; 1];
        let mut seen = String::new();
        while reader.read(&mut one).unwrap() > 0 {
            seen.push(one[0] as char);
        }
        assert_eq!(seen, "abcd");
    }

    #[test]
    fn decodes_chunked_bodies() {
        let raw = "4\r\nWiki\r\n5\r\npedia\r\n0\r\n\r\n";
        let mut body = body_of(raw);
        let mut out = String::new();
        body.read_to_string(&mut out).unwrap();
        assert_eq!(out, "Wikipedia");
    }

    #[test]
    fn decodes_chunked_bodies_with_small_reads() {
        // Small output buffers exercise the "chunk larger than buffer" path.
        let raw = "a\r\n0123456789\r\n3\r\nabc\r\n0\r\n\r\n";
        let mut body = body_of(raw);
        let mut out = Vec::new();
        let mut buf = [0u8; 3];
        loop {
            let n = body.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&buf[..n]);
        }
        assert_eq!(String::from_utf8(out).unwrap(), "0123456789abc");
    }

    #[test]
    fn short_circuits_when_chunk_larger_than_buffer() {
        let raw = "5\r\nhello\r\n0\r\n\r\n";
        let mut body = body_of(raw);
        let mut buf = [0u8; 2];
        let n = body.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"he");
    }

    #[test]
    fn decodes_content_length_bodies() {
        let mut body = Body::new(Cursor::new(b"{\"a\":1}".to_vec()), Framing::Length(7));
        let mut out = String::new();
        body.read_to_string(&mut out).unwrap();
        assert_eq!(out, "{\"a\":1}");
    }

    #[test]
    fn decodes_until_eof_bodies() {
        let mut body = Body::new(Cursor::new(b"line1\nline2\n".to_vec()), Framing::UntilEof);
        let mut out = String::new();
        body.read_to_string(&mut out).unwrap();
        assert_eq!(out, "line1\nline2\n");
    }

    #[test]
    fn parses_status_line_and_headers() {
        let raw = "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nX-Test: 1 \r\n\r\n";
        let mut r = BufReader::new(Cursor::new(raw.as_bytes().to_vec()));
        let (status, headers) = read_head(&mut r).unwrap();
        assert_eq!(status, 200);
        assert_eq!(
            headers[0],
            ("content-type".into(), "application/x-ndjson".into())
        );
        assert_eq!(headers[1], ("x-test".into(), "1".into()));
    }

    #[test]
    fn rejects_non_http_response() {
        let raw = "GARBAGE\r\n\r\n";
        let mut r = BufReader::new(Cursor::new(raw.as_bytes().to_vec()));
        assert!(matches!(read_head(&mut r), Err(HttpError::Protocol(_))));
    }
}
