import time

def remove_image_extensions(text):
    text = text.replace(".jpg", "")
    text = text.replace(".png", "")
    return text

def block_timer(name=""):
    class TimerContextManager:
        def __init__(self, name):
            self.name = name
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            self.end = time.perf_counter()
            print(f"{self.name} Time Cost: {(self.end - self.start) / 3600:.4f} hours")
            
    return TimerContextManager(name)