//! 画面（editor）を実行ファイルに埋め込んで配る。v2.0 の `editor/` をそのまま使う。

use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "editor/"]
struct Editor;

pub fn index_html() -> String {
    Editor::get("index.html")
        .map(|f| String::from_utf8_lossy(&f.data).into_owned())
        .unwrap_or_default()
}

/// 配ってよいのは `app.js` と `style.css` だけ。`(中身, Content-Type)`。
pub fn editor_file(name: &str) -> Option<(Vec<u8>, &'static str)> {
    let mime = match name {
        "app.js" => "text/javascript; charset=utf-8",
        "style.css" => "text/css; charset=utf-8",
        _ => return None,
    };
    Editor::get(name).map(|f| (f.data.into_owned(), mime))
}
