"""Hide this Tk player's cursor after 3 seconds; clicks reveal it again."""
import tkinter as tk
import weakref


class CursorAutoHide:
    def __init__(self, window, milliseconds=3000):
        self.window, self.milliseconds = window, milliseconds
        self.widgets = weakref.WeakKeyDictionary()
        self.tag = 'MemeCursorFirst' + str(id(self))
        self.job = None
        self.hidden = False
        self.closed = False
        # Our tag runs before widget handlers, including sleep's 'break' handler.
        window.bind_class(self.tag, '<ButtonPress>', self.show)
        self.map_binding = window.bind('<Map>', self.mapped, add='+')
        window.bind('<Destroy>', self.destroyed, add='+')
        self.track_tree(window)
        self.show()

    def track(self, widget):
        try:
            if widget.winfo_toplevel() is not self.window or widget in self.widgets:
                return
            self.widgets[widget] = widget.cget('cursor')
            widget.bindtags((self.tag,) + widget.bindtags())
            if self.hidden:
                widget.configure(cursor='none')
        except tk.TclError:
            pass

    def track_tree(self, widget):
        self.track(widget)
        for child in widget.winfo_children():
            self.track_tree(child)

    def mapped(self, event):
        if not self.closed:
            self.track_tree(event.widget)

    def show(self, event=None):
        if self.closed:
            return
        self.hidden = False
        for widget, cursor in list(self.widgets.items()):
            try:
                widget.configure(cursor=cursor)
            except tk.TclError:
                pass
        if self.job is not None:
            self.window.after_cancel(self.job)
        self.job = self.window.after(self.milliseconds, self.hide)

    def hide(self):
        self.job = None
        if self.closed:
            return
        self.hidden = True
        for widget in list(self.widgets):
            try:
                widget.configure(cursor='none')
            except tk.TclError:
                pass

    def destroyed(self, event):
        if event.widget is self.window:
            self.closed = True
            if self.job is not None:
                self.window.after_cancel(self.job)
                self.job = None
            self.window.unbind_class(self.tag, '<ButtonPress>')
