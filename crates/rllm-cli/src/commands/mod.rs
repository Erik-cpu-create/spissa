pub mod benchmark;
pub mod chat_session;
pub mod chat_session_token;
pub mod doctor;
pub mod import;
pub mod inspect;
pub mod pack;
pub mod run;
pub mod unpack;
pub mod verify;

pub mod common {
    use anyhow::Result;

    /// Parse size string like "1mb", "256kb", "4mb" into bytes
    pub fn parse_size(s: &str) -> Result<usize> {
        let s = s.trim().to_lowercase();

        if let Some(num) = s.strip_suffix("kb") {
            let n: usize = num.trim().parse()?;
            Ok(n * 1024)
        } else if let Some(num) = s.strip_suffix("mb") {
            let n: usize = num.trim().parse()?;
            Ok(n * 1024 * 1024)
        } else if let Some(num) = s.strip_suffix("gb") {
            let n: usize = num.trim().parse()?;
            Ok(n * 1024 * 1024 * 1024)
        } else if let Some(num) = s.strip_suffix('b') {
            let n: usize = num.trim().parse()?;
            Ok(n)
        } else {
            // Assume bytes if no suffix
            let n: usize = s.parse()?;
            Ok(n)
        }
    }

    #[test]
    fn test_parse_size() {
        assert_eq!(parse_size("1024").unwrap(), 1024);
        assert_eq!(parse_size("1kb").unwrap(), 1024);
        assert_eq!(parse_size("1mb").unwrap(), 1024 * 1024);
        assert_eq!(parse_size("1gb").unwrap(), 1024 * 1024 * 1024);
        assert_eq!(parse_size("256kb").unwrap(), 256 * 1024);
    }
}
