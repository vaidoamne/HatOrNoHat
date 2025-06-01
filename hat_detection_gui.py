import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from hat_detection import HatDetector
import os
import threading
import time
import datetime

# Define theme colors
DARK_BG = "#1E1E1E"
DARKER_BG = "#121212"
TEXT_COLOR = "#E0E0E0"
HIGHLIGHT_COLOR = "#FF7D00"  # Orange highlight
ACCENT_BG = "#2A2A2A"
INFO_BG = "#252525"
BANNER_BG = "#101010"

class HatDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hat Detection App")
        
        # Make the window start maximized
        self.root.state('zoomed')  # For Windows
        # Alternative methods for other platforms
        # self.root.attributes('-zoomed', True)  # For Linux
        # self.root.attributes('-fullscreen', True)  # Fullscreen option
        
        self.root.configure(bg=DARK_BG)
        self.root.minsize(900, 600)  # Set minimum size
        
        # Detection metrics
        self.detection_time = 0
        self.confidence_score = 0
        self.best_model_accuracy = 0
        
        # If a best model exists, try to read its accuracy from a metadata file
        if os.path.exists('best_model_accuracy.txt'):
            try:
                with open('best_model_accuracy.txt', 'r') as f:
                    self.best_model_accuracy = float(f.read().strip())
            except:
                self.best_model_accuracy = 97.90  # Default from the training output
        else:
            # Use the validation accuracy from training output
            self.best_model_accuracy = 97.90
        
        # Set up the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HatDetector()
        
        # Load the saved model if it exists
        if os.path.exists('best_model.pth'):
            self.model.load_state_dict(torch.load('best_model.pth', map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        else:
            messagebox.showerror("Error", "Model file 'best_model.pth' not found!")
        
        # Define image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create main layout with two panels
        self.setup_layout()
        
        # Image placeholder
        self.image_path = None
        self.display_image = None
        self.is_processing = False
        self.detection_count = 0
    
    def setup_layout(self):
        # Main container that will hold everything
        self.main_container = tk.Frame(self.root, bg=DARK_BG)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create content frame that will hold left and right panels
        self.content_frame = tk.Frame(self.main_container, bg=DARK_BG)
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for main content
        self.left_panel = tk.Frame(self.content_frame, bg=DARK_BG, padx=20, pady=20)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right panel for stats
        self.right_panel = tk.Frame(self.content_frame, bg=INFO_BG, width=250, padx=15, pady=20)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_panel.pack_propagate(False)  # Keep width fixed
        
        # Create banner at the bottom
        self.banner_frame = tk.Frame(self.main_container, bg=BANNER_BG, height=40)
        self.banner_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add the tagline to the banner
        tagline = tk.Label(
            self.banner_frame,
            text="Hat or no hat? That is the question",
            font=("Arial", 12, "italic"),
            bg=BANNER_BG,
            fg=HIGHLIGHT_COLOR,
            pady=10
        )
        tagline.pack()
        
        # Create widgets in each panel
        self.create_left_panel_widgets()
        self.create_right_panel_widgets()
    
    def create_right_panel_widgets(self):
        # Title for info panel
        info_title = tk.Label(
            self.right_panel,
            text="DETECTION STATS",
            font=("Arial", 14, "bold"),
            bg=INFO_BG,
            fg=HIGHLIGHT_COLOR
        )
        info_title.pack(pady=(0, 20))
        
        # Separator
        separator = ttk.Separator(self.right_panel, orient='horizontal')
        separator.pack(fill=tk.X, pady=5)
        
        # Model info section
        model_section = tk.Frame(self.right_panel, bg=INFO_BG, pady=10)
        model_section.pack(fill=tk.X)
        
        model_title = tk.Label(
            model_section,
            text="Model Information",
            font=("Arial", 12, "bold"),
            bg=INFO_BG,
            fg=TEXT_COLOR
        )
        model_title.pack(anchor=tk.W)
        
        # Device info
        device_frame = tk.Frame(model_section, bg=INFO_BG, pady=5)
        device_frame.pack(fill=tk.X)
        
        device_label = tk.Label(
            device_frame,
            text="Device:",
            font=("Arial", 10),
            bg=INFO_BG,
            fg=TEXT_COLOR,
            width=15,
            anchor=tk.W
        )
        device_label.pack(side=tk.LEFT)
        
        self.device_value = tk.Label(
            device_frame,
            text=str(self.device).upper(),
            font=("Arial", 10, "bold"),
            bg=INFO_BG,
            fg=HIGHLIGHT_COLOR
        )
        self.device_value.pack(side=tk.LEFT)
        
        # Model accuracy
        accuracy_frame = tk.Frame(model_section, bg=INFO_BG, pady=5)
        accuracy_frame.pack(fill=tk.X)
        
        accuracy_label = tk.Label(
            accuracy_frame,
            text="Model Accuracy:",
            font=("Arial", 10),
            bg=INFO_BG,
            fg=TEXT_COLOR,
            width=15,
            anchor=tk.W
        )
        accuracy_label.pack(side=tk.LEFT)
        
        self.accuracy_value = tk.Label(
            accuracy_frame,
            text=f"{self.best_model_accuracy:.2f}%",
            font=("Arial", 10, "bold"),
            bg=INFO_BG,
            fg=HIGHLIGHT_COLOR
        )
        self.accuracy_value.pack(side=tk.LEFT)
        
        # Separator
        separator2 = ttk.Separator(self.right_panel, orient='horizontal')
        separator2.pack(fill=tk.X, pady=10)
        
        # Current detection section
        detection_section = tk.Frame(self.right_panel, bg=INFO_BG, pady=10)
        detection_section.pack(fill=tk.X)
        
        detection_title = tk.Label(
            detection_section,
            text="Current Detection",
            font=("Arial", 12, "bold"),
            bg=INFO_BG,
            fg=TEXT_COLOR
        )
        detection_title.pack(anchor=tk.W)
        
        # Confidence score
        confidence_frame = tk.Frame(detection_section, bg=INFO_BG, pady=5)
        confidence_frame.pack(fill=tk.X)
        
        confidence_label = tk.Label(
            confidence_frame,
            text="Confidence:",
            font=("Arial", 10),
            bg=INFO_BG,
            fg=TEXT_COLOR,
            width=15,
            anchor=tk.W
        )
        confidence_label.pack(side=tk.LEFT)
        
        self.confidence_value = tk.Label(
            confidence_frame,
            text="-- %",
            font=("Arial", 10, "bold"),
            bg=INFO_BG,
            fg=HIGHLIGHT_COLOR
        )
        self.confidence_value.pack(side=tk.LEFT)
        
        # Processing time
        time_frame = tk.Frame(detection_section, bg=INFO_BG, pady=5)
        time_frame.pack(fill=tk.X)
        
        time_label = tk.Label(
            time_frame,
            text="Process Time:",
            font=("Arial", 10),
            bg=INFO_BG,
            fg=TEXT_COLOR,
            width=15,
            anchor=tk.W
        )
        time_label.pack(side=tk.LEFT)
        
        self.time_value = tk.Label(
            time_frame,
            text="-- ms",
            font=("Arial", 10, "bold"),
            bg=INFO_BG,
            fg=HIGHLIGHT_COLOR
        )
        self.time_value.pack(side=tk.LEFT)
        
        # Total detections
        count_frame = tk.Frame(detection_section, bg=INFO_BG, pady=5)
        count_frame.pack(fill=tk.X)
        
        count_label = tk.Label(
            count_frame,
            text="Total Detections:",
            font=("Arial", 10),
            bg=INFO_BG,
            fg=TEXT_COLOR,
            width=15,
            anchor=tk.W
        )
        count_label.pack(side=tk.LEFT)
        
        self.count_value = tk.Label(
            count_frame,
            text="0",
            font=("Arial", 10, "bold"),
            bg=INFO_BG,
            fg=HIGHLIGHT_COLOR
        )
        self.count_value.pack(side=tk.LEFT)
        
        # Separator
        separator3 = ttk.Separator(self.right_panel, orient='horizontal')
        separator3.pack(fill=tk.X, pady=10)
        
        # Time stamp section
        timestamp_section = tk.Frame(self.right_panel, bg=INFO_BG, pady=10)
        timestamp_section.pack(fill=tk.X)
        
        timestamp_title = tk.Label(
            timestamp_section,
            text="Last Detection",
            font=("Arial", 12, "bold"),
            bg=INFO_BG,
            fg=TEXT_COLOR
        )
        timestamp_title.pack(anchor=tk.W)
        
        # Timestamp
        self.timestamp_value = tk.Label(
            timestamp_section,
            text="--",
            font=("Arial", 10),
            bg=INFO_BG,
            fg=TEXT_COLOR,
            wraplength=220,
            justify=tk.LEFT
        )
        self.timestamp_value.pack(anchor=tk.W, pady=5)
        
        # Style the separators
        style = ttk.Style()
        style.configure("TSeparator", background=HIGHLIGHT_COLOR)
    
    def create_left_panel_widgets(self):
        # Title with custom styling
        title_label = tk.Label(
            self.left_panel, 
            text="HAT DETECTION", 
            font=("Arial", 22, "bold"),
            bg=DARK_BG,
            fg=HIGHLIGHT_COLOR
        )
        title_label.pack(pady=(0, 20))
        
        # Instructions with styled text
        instructions = tk.Label(
            self.left_panel, 
            text="Upload an image to detect if the person is wearing a hat",
            font=("Arial", 12),
            bg=DARK_BG,
            fg=TEXT_COLOR
        )
        instructions.pack(pady=(0, 20))
        
        # Frame for the image with border styling
        self.image_container = tk.Frame(
            self.left_panel, 
            bd=2, 
            relief=tk.GROOVE, 
            bg=DARKER_BG,
            highlightbackground=HIGHLIGHT_COLOR,
            highlightthickness=1
        )
        self.image_container.pack(pady=15)
        
        # Image frame with fixed size
        self.image_frame = tk.Frame(
            self.image_container,
            width=350, 
            height=350, 
            bg=DARKER_BG
        )
        self.image_frame.pack(padx=10, pady=10)
        self.image_frame.pack_propagate(False)  # Keep the frame size fixed
        
        # Image placeholder with styling
        self.image_label = tk.Label(
            self.image_frame, 
            text="Image will appear here", 
            bg=DARKER_BG,
            fg=TEXT_COLOR,
            font=("Arial", 12, "italic")
        )
        self.image_label.pack(expand=True, fill=tk.BOTH)
        
        # Button frame
        button_frame = tk.Frame(self.left_panel, bg=DARK_BG)
        button_frame.pack(pady=15)
        
        # Style for buttons
        button_style = {
            "bg": ACCENT_BG,
            "fg": TEXT_COLOR,
            "activebackground": HIGHLIGHT_COLOR,
            "activeforeground": DARKER_BG,
            "font": ("Arial", 11),
            "bd": 0,
            "padx": 15,
            "pady": 8,
            "width": 15,
            "cursor": "hand2"
        }
        
        # Upload button with custom styling
        upload_button = tk.Button(
            button_frame, 
            text="Upload Image", 
            command=self.upload_image,
            **button_style
        )
        upload_button.pack(side=tk.LEFT, padx=10)
        
        # Detect button with custom styling
        self.detect_button = tk.Button(
            button_frame, 
            text="Detect Hat", 
            command=self.start_detection,
            **button_style
        )
        self.detect_button.pack(side=tk.LEFT, padx=10)
        
        # Progress frame
        self.progress_frame = tk.Frame(self.left_panel, bg=DARK_BG)
        self.progress_frame.pack(pady=(5, 15), fill=tk.X)
        
        # Progress bar (hidden initially)
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(
            self.progress_frame, 
            variable=self.progress_var,
            length=350, 
            mode='indeterminate'
        )
        self.progress.pack(pady=5)
        self.progress.pack_forget()  # Hide initially
        
        # Processing label (hidden initially)
        self.processing_label = tk.Label(
            self.progress_frame,
            text="Processing image...",
            bg=DARK_BG,
            fg=HIGHLIGHT_COLOR,
            font=("Arial", 10, "italic")
        )
        self.processing_label.pack()
        self.processing_label.pack_forget()  # Hide initially
        
        # Result label with styled text
        self.result_frame = tk.Frame(self.left_panel, bg=ACCENT_BG, bd=1, relief=tk.FLAT)
        self.result_frame.pack(pady=10, fill=tk.X)
        self.result_frame.pack_forget()  # Hide initially
        
        self.result_label = tk.Label(
            self.result_frame,
            text="", 
            font=("Arial", 16, "bold"),
            bg=ACCENT_BG,
            fg=HIGHLIGHT_COLOR,
            padx=20,
            pady=15
        )
        self.result_label.pack()
        
        # Configure the progress bar style
        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            "TProgressbar",
            thickness=10,
            troughcolor=ACCENT_BG,
            background=HIGHLIGHT_COLOR
        )
    
    def upload_image(self):
        # Reset detection state
        self.hide_result()
        
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            self.image_path = file_path
            # Load and display the image
            image = Image.open(file_path)
            
            # Calculate resize dimensions while maintaining aspect ratio
            width, height = image.size
            max_size = 350
            
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
            self.display_image = ImageTk.PhotoImage(image)
            
            # Position image centered in frame
            self.image_label.config(image=self.display_image, text="")
    
    def start_detection(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please upload an image first!")
            return
        
        if self.is_processing:
            return  # Prevent multiple detection processes
        
        # Update UI for processing state
        self.is_processing = True
        self.detect_button.config(state=tk.DISABLED)
        self.progress.pack(pady=5)
        self.processing_label.pack()
        self.progress.start(10)  # Start progress animation
        
        # Reset detection stats
        self.time_value.config(text="-- ms")
        self.confidence_value.config(text="-- %")
        
        # Start detection in a separate thread
        threading.Thread(target=self.detect_hat).start()
    
    def detect_hat(self):
        start_time = time.time()
        
        try:
            # Load and preprocess the image
            image = Image.open(self.image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                _, prediction = torch.max(outputs, 1)
                
                # Get confidence score (softmax of outputs)
                softmax = torch.nn.functional.softmax(outputs, dim=1)
                confidence = softmax[0][prediction.item()].item() * 100
            
            # Calculate detection time
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Store detection metrics
            self.detection_time = elapsed_time
            self.confidence_score = confidence
            self.detection_count += 1
            
            # Get current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Display result (in main thread)
            self.root.after(0, lambda: self.update_result(prediction.item(), confidence, elapsed_time, timestamp))
                
        except Exception as e:
            self.root.after(0, lambda: self.show_error(str(e)))
        finally:
            # Reset processing state (in main thread)
            self.root.after(0, self.reset_ui)
    
    def update_result(self, prediction, confidence, elapsed_time, timestamp):
        # Show result frame
        self.result_frame.pack(pady=10, fill=tk.X)
        
        # Display result with confidence
        if prediction == 1:
            result = f"Hat Detected!"
            self.result_label.config(text=result, fg=HIGHLIGHT_COLOR)
        else:
            result = f"No Hat Detected"
            self.result_label.config(text=result, fg=TEXT_COLOR)
        
        # Update stats in right panel
        self.confidence_value.config(text=f"{confidence:.2f}%")
        self.time_value.config(text=f"{elapsed_time:.2f} ms")
        self.count_value.config(text=str(self.detection_count))
        self.timestamp_value.config(text=timestamp)
    
    def reset_ui(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.processing_label.pack_forget()
        self.detect_button.config(state=tk.NORMAL)
        self.is_processing = False
    
    def hide_result(self):
        self.result_frame.pack_forget()
    
    def show_error(self, error_msg):
        messagebox.showerror("Error", f"An error occurred: {error_msg}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HatDetectionGUI(root)
    root.mainloop() 