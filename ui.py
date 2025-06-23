from PIL import Image, ImageTk, ImageSequence
import customtkinter as ctk
import webbrowser
import re
import sys, os
import traceback
import threading
from datetime import datetime
from rag_chain import ask_bot, detect_intent

def resource_path(relative_path):
    """ Get absolute path to resource (for PyInstaller compatibility) """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


class ChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Silvestre Assistant")
        self.geometry("600x600")
        self.resizable(False, False) 
        ctk.set_appearance_mode("light")
        self.chat_history = []

        # Chat area
        self.chat_frame = ctk.CTkFrame(self, fg_color="white")
        self.chat_frame.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        self.canvas = ctk.CTkCanvas(self.chat_frame, bg="white", highlightthickness=0)
        self.scrollable_frame = ctk.CTkFrame(self.canvas, fg_color="white")

        self.scrollbar = ctk.CTkScrollbar(self.chat_frame, command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Entry and buttons
        self.entry_frame = ctk.CTkFrame(self)
        self.entry_frame.pack(fill="x", padx=10, pady=10)

        self.entry = ctk.CTkEntry(self.entry_frame, placeholder_text="Type your message here...")
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.entry.bind("<Return>", self.send_message)

        self.send_btn = ctk.CTkButton(self.entry_frame, text="Send", command=self.send_message)
        self.send_btn.pack(side="right")

        self.refresh_btn = ctk.CTkButton(self.entry_frame, text="Refresh Data", command=self.refresh_data)
        self.refresh_btn.pack(side="left", padx=(0, 10))

    def refresh_data(self):
        from db import refresh_database
        from rag_chain import rebuild_vectorstore

        def run_refresh():
            self.refresh_btn.configure(state="disabled")
            self.show_spinner()
            self.add_bubble("🔄 Refreshing knowledge base. Please wait...", sender="bot")

            try:
                refresh_database()
                rebuild_vectorstore()
                self.add_bubble("✅ Knowledge base refreshed successfully!", sender="bot")
            except Exception as e:
                self.add_bubble(f"❌ Refresh failed: {e}", sender="bot")
            finally:
                self.hide_spinner()
                self.refresh_btn.configure(state="normal")

        threading.Thread(target=run_refresh, daemon=True).start()


    def send_message(self, event=None):
        user_msg = self.entry.get().strip()
        if not user_msg:
            return

        self.add_bubble(user_msg, "user")
        self.update_idletasks()
        self.entry.delete(0, "end")
        self.entry.configure(state="disabled")
        self.send_btn.configure(state="disabled")

        # Show placeholder while bot is typing
        typing_bubble = self.add_bubble("Typing...", "bot")

        def run_response():
            try:
                response = ask_bot(user_msg, self.chat_history)
                self.chat_history.append({"role": "user", "content": user_msg})
                self.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                error_details = traceback.format_exc()
                print("[ERROR]", error_details)
                response = f"[ERROR] {str(e) or 'Unknown error. Check terminal.'}"

            # Remove "Typing..." and replace with actual response
            def update_ui():
                if typing_bubble.winfo_exists():
                    typing_bubble.destroy()
                self.add_bubble(response, "bot")
                self.entry.configure(state="normal")
                self.send_btn.configure(state="normal")

            self.after(0, update_ui)


        threading.Thread(target=run_response, daemon=True).start()

    def add_bubble(self, msg, sender):
        bubble_color = "#0084FF" if sender == "user" else "#E4E6EB"
        text_color = "white" if sender == "user" else "black"
        anchor = "e" if sender == "user" else "w"
        justify = "right" if sender == "user" else "left"

        outer = ctk.CTkFrame(self.scrollable_frame, fg_color="white")
        outer.pack(fill="x", pady=4)

        inner = ctk.CTkFrame(outer, fg_color="white")
        if sender == "user":
            inner.pack(anchor="e", padx=(0, 10))
        else:
            inner.pack(anchor="w", padx=(10, 120))

        bubble = ctk.CTkFrame(inner, fg_color=bubble_color, corner_radius=18)
        bubble.pack(anchor=anchor, padx=4, pady=2)
        bubble.configure(width=420)

        parts = re.split(r'(https?://\S+)', msg)
        for part in parts:
            if re.match(r'https?://', part):
                cleaned_url = part.rstrip('.,)]')
                lbl = ctk.CTkLabel(
                    bubble,
                    text=cleaned_url,
                    text_color="#1a0dab",
                    font=ctk.CTkFont(underline=True),
                    cursor="hand2",
                    fg_color="transparent",
                    wraplength=400,
                    anchor="w",
                    justify="left"
                )
                lbl.pack(anchor="w", padx=12, pady=2, fill="x")
                lbl.bind("<Button-1>", lambda e, url=cleaned_url: webbrowser.open(url))
            else:
                lbl = ctk.CTkLabel(
                    bubble,
                    text=part.strip(),
                    text_color=text_color,
                    fg_color="transparent",
                    anchor="w",
                    justify="left",
                    wraplength=400
                )
                lbl.pack(anchor="w", padx=12, pady=2, fill="x")

        timestamp = datetime.now().strftime("%I:%M %p")
        ts_label = ctk.CTkLabel(
            inner,
            text=timestamp,
            text_color="#888888",
            font=ctk.CTkFont(size=10),
            fg_color="transparent"
        )
        ts_label.pack(anchor=anchor, pady=(2, 0))

        self.update_idletasks()
        self.smooth_scroll_to_bottom()
        return outer


    def smooth_scroll_to_bottom(self, steps=10, delay=10):
        """
        Smoothly scrolls to the bottom over `steps` increments.
        """
        current = self.canvas.yview()[0]  # current top position (0.0 to 1.0)
        target = 1.0
        step_size = (target - current) / steps

        def scroll_step(i=0):
            if i < steps:
                new_pos = current + step_size * i
                self.canvas.yview_moveto(new_pos)
                self.after(delay, lambda: scroll_step(i + 1))
            else:
                self.canvas.yview_moveto(1.0)

        scroll_step()



    def show_spinner(self):
    # Overlay with blur effect
        self.overlay = ctk.CTkFrame(self, fg_color="#ffffff")  # Semi-transparent white
        self.overlay.place(relx=0.5, rely=0.5, anchor="center", relwidth=1, relheight=1)

    # Spinner frames
        self.spinner_image = Image.open(resource_path("assets/spinner.gif"))
        self.spinner_frames = [
            ImageTk.PhotoImage(frame.copy().resize((48, 48)).convert("RGBA"))
            for frame in ImageSequence.Iterator(self.spinner_image)
        ]

    # Spinner
        self.spinner_label = ctk.CTkLabel(self.overlay, text="", image=self.spinner_frames[0], fg_color="transparent")
        self.spinner_label.pack(pady=(200, 5))

    # Loading text
        self.loading_text = ctk.CTkLabel(self.overlay, text="Loading...", text_color="#333333", font=ctk.CTkFont(size=14))
        self.loading_text.pack()

        self.spinner_running = True
        self.spinner_index = 0
        self.animate_spinner_frame()




    def animate_spinner_frame(self):
        if self.spinner_running:
            frame = self.spinner_frames[self.spinner_index]
            self.spinner_label.configure(image=frame)
            self.spinner_index = (self.spinner_index + 1) % len(self.spinner_frames)
            self.after(100, self.animate_spinner_frame)

    def hide_spinner(self):
        self.spinner_running = False
        if hasattr(self, 'spinner_label'):
            self.spinner_label.destroy()

        if hasattr(self, 'loading_text'):
            self.loading_text.destroy()

    # Show ✅ checkmark
        check_img = Image.open(resource_path("assets/check.png")).resize((48, 48))
        self.check_image = ImageTk.PhotoImage(check_img)

        self.check_label = ctk.CTkLabel(self.overlay, text="", image=self.check_image, fg_color="transparent")
        self.check_label.pack(pady=(200, 5))

    # Completion text
        self.done_label = ctk.CTkLabel(self.overlay, text="Refresh Complete!", text_color="#28a745", font=ctk.CTkFont(size=14, weight="bold"))
        self.done_label.pack()

    # Remove overlay after delay
        self.after(1500, lambda: self.overlay.destroy())



