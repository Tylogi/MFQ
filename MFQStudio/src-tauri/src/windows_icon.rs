use tauri::{image::Image, WebviewWindow};

#[link(name = "user32")]
extern "system" {
    fn SendMessageW(window: isize, message: u32, wparam: usize, lparam: isize) -> isize;
}

pub fn set(window: &WebviewWindow) -> tauri::Result<()> {
    window.set_icon(Image::new(
        include_bytes!(concat!(env!("OUT_DIR"), "/window-icon.rgba")),
        256,
        256,
    ))?;
    let handle = window.hwnd()?.0 as isize;
    // Tauri updates ICON_SMALL only. Reuse its owned high-resolution icon for
    // ICON_BIG too, after the update has run on the window thread.
    window.run_on_main_thread(move || unsafe {
        const WM_GETICON: u32 = 0x007f;
        const WM_SETICON: u32 = 0x0080;
        const ICON_SMALL: usize = 0;
        const ICON_BIG: usize = 1;
        let icon = SendMessageW(handle, WM_GETICON, ICON_SMALL, 0);
        if icon != 0 {
            SendMessageW(handle, WM_SETICON, ICON_BIG, icon);
        }
    })
}
