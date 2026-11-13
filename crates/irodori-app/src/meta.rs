//! `/api/meta`：絵文字パレット（上流の一覧を v2.0 のグループ順で）と、クレジット・注意事項の HTML。
//!
//! データは `tools/compat/gen_meta.py` が v2.0 と上流の実物から作る。クレジット（制作者表記・ライセンス・禁止事項）は
//! 画面から消してはならない。

use std::sync::OnceLock;

use axum::Json;
use pulldown_cmark::{html, Event, Options, Parser};
use serde_json::{json, Value};

const PALETTE: &str = include_str!("../data/emoji_palette.json");
const CREDITS_MD: &str = include_str!("../data/credits.md");

/// CommonMark を HTML にする。Markdown 中の生の HTML は無効（文字として表示する）。
pub fn render_markdown(markdown: &str) -> String {
    let events = Parser::new_ext(markdown, Options::empty()).map(|event| match event {
        Event::Html(raw) | Event::InlineHtml(raw) => Event::Text(raw),
        other => other,
    });
    let mut out = String::new();
    html::push_html(&mut out, events);
    out
}

fn meta_value() -> &'static Value {
    static META: OnceLock<Value> = OnceLock::new();
    META.get_or_init(|| {
        let palette: Value = serde_json::from_str(PALETTE).expect("emoji_palette.json is valid");
        json!({
            "emojis": palette["emojis"],
            "emoji_groups": palette["emoji_groups"],
            "credits": render_markdown(CREDITS_MD),
        })
    })
}

pub async fn meta() -> Json<Value> {
    Json(meta_value().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_html_in_markdown_is_shown_as_text() {
        let html = render_markdown("<script>alert(1)</script>\n\ntext <b>bold</b>");
        assert!(
            !html.contains("<script>") && !html.contains("<b>"),
            "{html}"
        );
        assert!(html.contains("&lt;script&gt;"));
    }

    #[test]
    fn markdown_links_and_headings_render() {
        let html = render_markdown("### 見出し\n[リンク](https://example.com)");
        assert!(
            html.contains("<h3>見出し</h3>")
                && html.contains(r#"<a href="https://example.com">リンク</a>"#),
            "{html}"
        );
    }
}
