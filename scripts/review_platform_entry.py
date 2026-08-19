from __future__ import annotations

from cellvision.review_platform import main


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Cell Vision Review", f"无法打开审核数据包：\n{type(exc).__name__}: {exc}")
            root.destroy()
        finally:
            raise
