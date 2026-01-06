# ============================================================================
# COMPREHENSIVE FACE RECOGNITION SYSTEM
# ============================================================================
import cv2
import numpy as np
import face_recognition
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import argparse
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self, tolerance: float = 0.6):
        """
        Initialize Face Recognition System

        Args:
            tolerance: Distance tolerance for face matching (lower = stricter)
                      Recommended values: 0.4-0.7
        """
        self.tolerance = tolerance
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_images = {}

    # ============================================================================
    # IMAGE UTILITIES
    # ============================================================================

    def resize_image(self, image: np.ndarray, max_dimension: int = 800) -> np.ndarray:
        """
        Resize image maintaining aspect ratio for faster processing

        Args:
            image: Input image
            max_dimension: Maximum dimension (width or height)

        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            return cv2.resize(image, (new_w, new_h))
        return image

    def load_and_preprocess_image(self, image_path: str,
                                 max_dimension: int = 800) -> Optional[np.ndarray]:
        """
        Load and preprocess image with error handling

        Args:
            image_path: Path to image file
            max_dimension: Maximum dimension for resizing

        Returns:
            Preprocessed RGB image or None if error
        """
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found: {image_path}")
            return None

        try:
            # Load image
            image = face_recognition.load_image_file(image_path)

            # Convert to RGB (face_recognition expects RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize for faster processing
            rgb_image = self.resize_image(rgb_image, max_dimension)

            return rgb_image

        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None

    # ============================================================================
    # FACE DETECTION & ENCODING
    # ============================================================================

    def detect_and_encode_faces(self, image: np.ndarray,
                               num_jitters: int = 1) -> Tuple[List, List]:
        """
        Detect faces and generate encodings

        Args:
            image: RGB image
            num_jitters: How many times to re-sample the face (higher = more accurate but slower)

        Returns:
            Tuple of (face_locations, face_encodings)
        """
        # Detect face locations
        face_locations = face_recognition.face_locations(image, model="hog")

        if len(face_locations) == 0:
            print("⚠️ No faces detected in the image!")
            return [], []

        # Generate face encodings
        face_encodings = face_recognition.face_encodings(
            image, face_locations, num_jitters=num_jitters
        )

        print(f"✅ Detected {len(face_locations)} face(s)")
        return face_locations, face_encodings

    # ============================================================================
    # DATABASE MANAGEMENT
    # ============================================================================

    def load_known_faces_from_directory(self, directory_path: str) -> bool:
        """
        Load multiple known faces from a directory

        Args:
            directory_path: Path to directory containing known face images

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(directory_path):
            print(f"❌ Directory not found: {directory_path}")
            return False

        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_images = {}

        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []

        for fmt in supported_formats:
            image_files.extend(Path(directory_path).glob(f"*{fmt}"))
            image_files.extend(Path(directory_path).glob(f"*{fmt.upper()}"))

        if not image_files:
            print(f"❌ No image files found in {directory_path}")
            return False

        print(f"📁 Found {len(image_files)} image files")

        for file_path in image_files:
            print(f"Processing: {file_path.name}...", end=" ")

            # Load and preprocess image
            image = self.load_and_preprocess_image(str(file_path))
            if image is None:
                print("Failed to load")
                continue

            # Detect and encode faces
            face_locations, face_encodings = self.detect_and_encode_faces(image)

            if len(face_encodings) == 0:
                print("No face found")
                continue
            elif len(face_encodings) > 1:
                print(f"Found {len(face_encodings)} faces, using first one")

            # Store first face encoding
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(file_path.stem)  # Use filename without extension as name
            self.known_face_images[file_path.stem] = image

            print(f"✅ Added as '{file_path.stem}'")

        print(f"\n✅ Successfully loaded {len(self.known_face_encodings)} known faces")
        return True

    def add_known_face(self, image_path: str, name: str) -> bool:
        """
        Add a single known face to the database

        Args:
            image_path: Path to face image
            name: Name to associate with the face

        Returns:
            True if successful, False otherwise
        """
        image = self.load_and_preprocess_image(image_path)
        if image is None:
            return False

        face_locations, face_encodings = self.detect_and_encode_faces(image)

        if len(face_encodings) == 0:
            print("❌ No face found in the image")
            return False

        self.known_face_encodings.append(face_encodings[0])
        self.known_face_names.append(name)
        self.known_face_images[name] = image

        print(f"✅ Added '{name}' to known faces database")
        return True

    # ============================================================================
    # ENCODINGS SAVE/LOAD
    # ============================================================================

    def save_encodings(self, filename: str = "face_encodings.pkl") -> bool:
        """
        Save face encodings to file

        Args:
            filename: Path to save file

        Returns:
            True if successful, False otherwise
        """
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'timestamp': datetime.now().isoformat()
            }

            with open(filename, 'wb') as f:
                pickle.dump(data, f)

            print(f"✅ Encodings saved to {filename}")
            return True

        except Exception as e:
            print(f"❌ Error saving encodings: {e}")
            return False

    def load_encodings(self, filename: str = "face_encodings.pkl") -> bool:
        """
        Load face encodings from file

        Args:
            filename: Path to load file

        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filename):
                print(f"❌ File not found: {filename}")
                return False

            with open(filename, 'rb') as f:
                data = pickle.load(f)

            self.known_face_encodings = data['encodings']
            self.known_face_names = data['names']

            if 'timestamp' in data:
                print(f"📅 Encodings created: {data['timestamp']}")

            print(f"✅ Loaded {len(self.known_face_encodings)} face encodings from {filename}")
            return True

        except Exception as e:
            print(f"❌ Error loading encodings: {e}")
            return False

    # ============================================================================
    # FACE COMPARISON
    # ============================================================================

    def compare_faces(self, test_image_path: str,
                     display_results: bool = True) -> List[Dict[str, Any]]:
        """
        Compare faces in test image with known faces

        Args:
            test_image_path: Path to test image
            display_results: Whether to display visual results

        Returns:
            List of comparison results
        """
        if not self.known_face_encodings:
            print("❌ No known faces loaded. Please load known faces first.")
            return []

        # Load test image
        test_image = self.load_and_preprocess_image(test_image_path)
        if test_image is None:
            return []

        # Detect faces in test image
        test_face_locations, test_face_encodings = self.detect_and_encode_faces(test_image)

        if len(test_face_encodings) == 0:
            print("❌ No faces found in test image")
            return []

        results = []

        for i, (face_encoding, face_location) in enumerate(zip(test_face_encodings, test_face_locations)):
            # Compare with all known faces
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )

            # Find best match
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            is_match = best_distance <= self.tolerance

            result = {
                'face_index': i,
                'location': face_location,
                'is_match': is_match,
                'best_match_name': self.known_face_names[best_match_index] if is_match else "Unknown",
                'best_match_distance': float(best_distance),
                'all_distances': [float(d) for d in face_distances],
                'confidence': float((1 - best_distance) * 100) if best_distance <= 1.0 else 0.0
            }

            results.append(result)

            print(f"\n🔍 Face {i+1} Results:")
            print(f"   Match: {is_match}")
            print(f"   Best Match: {result['best_match_name']}")
            print(f"   Distance: {best_distance:.4f}")
            print(f"   Confidence: {result['confidence']:.1f}%")
            print(f"   Location: {face_location}")

            # Display visual results
            if display_results:
                self.display_face_comparison(test_image, face_location, result)

        return results

    # ============================================================================
    # VISUALIZATION
    # ============================================================================

    def draw_face_rectangle(self, image: np.ndarray,
                           face_location: Tuple[int, int, int, int],
                           color: Tuple[int, int, int] = (0, 255, 0),
                           thickness: int = 2,
                           label: str = "") -> np.ndarray:
        """
        Draw rectangle around detected face with optional label

        Args:
            image: Input image
            face_location: (top, right, bottom, left)
            color: Rectangle color (BGR)
            thickness: Line thickness
            label: Text label to display above rectangle

        Returns:
            Image with rectangle drawn
        """
        top, right, bottom, left = face_location
        image_copy = image.copy()

        # Draw rectangle
        cv2.rectangle(image_copy, (left, top), (right, bottom), color, thickness)

        # Draw label background
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2

            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            # Draw filled rectangle for text background
            cv2.rectangle(
                image_copy,
                (left, top - text_height - 10),
                (left + text_width, top),
                color,
                cv2.FILLED
            )

            # Draw text
            cv2.putText(
                image_copy,
                label,
                (left, top - 5),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )

        return image_copy

    def display_face_comparison(self, test_image: np.ndarray,
                               face_location: Tuple[int, int, int, int],
                               result: Dict[str, Any]) -> None:
        """
        Display comparison results visually

        Args:
            test_image: Test image
            face_location: Location of detected face
            result: Comparison result dictionary
        """
        # Create a copy for display
        display_image = test_image.copy()

        # Determine color based on match result
        if result['is_match']:
            color = (0, 255, 0)  # Green for match
            status_text = f"MATCH: {result['best_match_name']}"
        else:
            color = (0, 0, 255)  # Red for no match
            status_text = "UNKNOWN"

        # Draw rectangle around face
        display_image = self.draw_face_rectangle(
            display_image, face_location, color, 3, status_text
        )

        # Add distance and confidence info
        info_text = f"Distance: {result['best_match_distance']:.4f} | Confidence: {result['confidence']:.1f}%"
        cv2.putText(
            display_image,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        # Add tolerance info
        tolerance_text = f"Tolerance: {self.tolerance}"
        cv2.putText(
            display_image,
            tolerance_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1
        )

        # Display
        cv2.imshow(f"Face Comparison - Face {result['face_index'] + 1}", display_image)
        print("\n🖼️ Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def display_comparison_grid(self, test_image_path: str,
                               known_image_path: str,
                               result: bool,
                               distance: float) -> None:
        """
        Display images side by side with comparison result

        Args:
            test_image_path: Path to test image
            known_image_path: Path to known image
            result: Comparison result
            distance: Face distance
        """
        test_image = self.load_and_preprocess_image(test_image_path)
        known_image = self.load_and_preprocess_image(known_image_path)

        if test_image is None or known_image is None:
            return

        # Resize images to same height for display
        max_height = max(test_image.shape[0], known_image.shape[0])

        def resize_to_height(img, height):
            h, w = img.shape[:2]
            scale = height / h
            new_w = int(w * scale)
            return cv2.resize(img, (new_w, height))

        test_display = resize_to_height(test_image, max_height)
        known_display = resize_to_height(known_image, max_height)

        # Create grid
        grid = np.hstack((known_display, test_display))

        # Add separator line
        cv2.line(
            grid,
            (known_display.shape[1], 0),
            (known_display.shape[1], max_height),
            (255, 255, 255),
            2
        )

        # Add result text
        result_text = "MATCH ✅" if result else "NO MATCH ❌"
        color = (0, 255, 0) if result else (0, 0, 255)

        cv2.putText(
            grid,
            result_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        # Add distance info
        distance_text = f"Distance: {distance:.4f}"
        confidence_text = f"Confidence: {(1 - distance) * 100:.1f}%"

        cv2.putText(
            grid,
            distance_text,
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            grid,
            confidence_text,
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Add labels
        cv2.putText(
            grid,
            "Known Face",
            (10, max_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1
        )

        cv2.putText(
            grid,
            "Test Face",
            (known_display.shape[1] + 10, max_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1
        )

        cv2.imshow("Face Comparison Grid", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ============================================================================
    # REAL-TIME RECOGNITION
    # ============================================================================

    def real_time_recognition(self, camera_index: int = 0,
                             display_scale: float = 0.25) -> None:
        """
        Real-time face recognition from webcam

        Args:
            camera_index: Camera device index
            display_scale: Scale factor for processing (smaller = faster but less accurate)
        """
        if not self.known_face_encodings:
            print("❌ No known faces loaded. Please load known faces first.")
            return

        video_capture = cv2.VideoCapture(camera_index)

        if not video_capture.isOpened():
            print("❌ Cannot open camera")
            return

        print("\n📹 Starting real-time face recognition...")
        print("   Press 'q' to quit")
        print("   Press 's' to save current frame")

        frame_count = 0
        process_this_frame = True

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("❌ Failed to grab frame")
                break

            # Only process every other frame to save processing time
            if process_this_frame:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)

                # Convert to RGB
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Detect faces
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []

                for face_encoding in face_encodings:
                    # Compare with known faces
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding, tolerance=self.tolerance
                    )
                    name = "Unknown"

                    if True in matches:
                        # Find the best match
                        face_distances = face_recognition.face_distance(
                            self.known_face_encodings, face_encoding
                        )
                        best_match_index = np.argmin(face_distances)
                        name = self.known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations
                top = int(top / display_scale)
                right = int(right / display_scale)
                bottom = int(bottom / display_scale)
                left = int(left / display_scale)

                # Choose color based on recognition result
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                # Draw rectangle
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Draw label
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

            # Add FPS counter
            frame_count += 1
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Add known faces count
            cv2.putText(frame, f"Known Faces: {len(self.known_face_names)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Display
            cv2.imshow('Real-time Face Recognition', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Frame saved as {filename}")

        video_capture.release()
        cv2.destroyAllWindows()
        print("\n✅ Real-time recognition stopped")

    # ============================================================================
    # UTILITIES
    # ============================================================================

    def print_database_info(self) -> None:
        """Print information about loaded face database"""
        print("\n📊 FACE DATABASE INFO")
        print("=" * 40)
        print(f"Total known faces: {len(self.known_face_names)}")
        print(f"Tolerance setting: {self.tolerance}")
        print("\nKnown faces:")
        for i, name in enumerate(self.known_face_names, 1):
            print(f"  {i}. {name}")
        print("=" * 40)

    def test_single_comparison(self, known_image_path: str,
                              test_image_path: str) -> Dict[str, Any]:
        """
        Simple one-to-one face comparison

        Args:
            known_image_path: Path to known face image
            test_image_path: Path to test face image

        Returns:
            Comparison results
        """
        print(f"\n🔍 Comparing: {Path(known_image_path).name} vs {Path(test_image_path).name}")
        print("=" * 50)

        # Load images
        known_image = self.load_and_preprocess_image(known_image_path)
        test_image = self.load_and_preprocess_image(test_image_path)

        if known_image is None or test_image is None:
            return {}

        # Detect and encode faces
        known_locations, known_encodings = self.detect_and_encode_faces(known_image)
        test_locations, test_encodings = self.detect_and_encode_faces(test_image)

        if not known_encodings or not test_encodings:
            return {}

        # Compare faces
        result = face_recognition.compare_faces([known_encodings[0]], test_encodings[0])
        distance = face_recognition.face_distance([known_encodings[0]], test_encodings[0])

        # Display results
        self.display_comparison_grid(
            test_image_path,
            known_image_path,
            result[0],
            float(distance[0])
        )

        return {
            'match': result[0],
            'distance': float(distance[0]),
            'confidence': float((1 - distance[0]) * 100)
        }


# ============================================================================
# MAIN FUNCTION WITH COMMAND LINE INTERFACE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument("--mode", choices=["compare", "realtime", "train", "test"],
                       default="compare", help="Operation mode")
    parser.add_argument("--known", type=str, help="Path to known image or directory")
    parser.add_argument("--test", type=str, help="Path to test image")
    parser.add_argument("--tolerance", type=float, default=0.6,
                       help="Face matching tolerance (default: 0.6)")
    parser.add_argument("--save", type=str, help="Save encodings to file")
    parser.add_argument("--load", type=str, help="Load encodings from file")

    args = parser.parse_args()

    # Initialize system
    face_system = FaceRecognitionSystem(tolerance=args.tolerance)

    # Determine actual mode based on arguments
    actual_mode = args.mode

    # Auto-detect mode if not specified but have required args
    if args.mode == "compare" and args.load and args.test:
        actual_mode = "compare"
    elif args.known and os.path.isdir(args.known):
        actual_mode = "train"
    elif args.test and (args.known or args.load):
        actual_mode = "compare"

    print(f"🚀 Mode: {actual_mode.upper()}")

    # Handle load/save options
    if args.load:
        if not face_system.load_encodings(args.load):
            print("❌ Failed to load encodings")
            return

    # Execute based on mode
    if actual_mode == "train" and args.known:
        print("📚 Building face database...")
        if os.path.isdir(args.known):
            face_system.load_known_faces_from_directory(args.known)
        else:
            name = Path(args.known).stem
            face_system.add_known_face(args.known, name)

        if args.save:
            face_system.save_encodings(args.save)

        face_system.print_database_info()

    elif actual_mode == "compare" and args.test:
        print("🔍 Comparing faces...")

        # Check if we have known faces loaded
        if not face_system.known_face_encodings:
            if args.known:
                if os.path.isdir(args.known):
                    face_system.load_known_faces_from_directory(args.known)
                else:
                    name = Path(args.known).stem
                    face_system.add_known_face(args.known, name)
            else:
                print("❌ Error: No known faces loaded. Use --load or --known")
                return

        results = face_system.compare_faces(args.test)

        if results:
            print("\n📊 SUMMARY:")
            print("=" * 40)
            for result in results:
                status = "✅ MATCH" if result['is_match'] else "❌ NO MATCH"
                print(f"Face {result['face_index'] + 1}: {status}")
                print(f"  Identity: {result['best_match_name']}")
                print(f"  Confidence: {result['confidence']:.1f}%")
                print(f"  Distance: {result['best_match_distance']:.4f}")
                print()

        face_system.print_database_info()

    elif actual_mode == "realtime":
        print("📹 Starting real-time recognition...")
        if not face_system.known_face_encodings and args.known:
            face_system.load_known_faces_from_directory(args.known)

        if not face_system.known_face_encodings:
            print("❌ No known faces loaded for real-time recognition")
            return

        face_system.real_time_recognition()

    elif actual_mode == "test":
        print("🧪 Running test mode...")
        demo_simple_comparison()

    else:
        print("❌ Invalid arguments or missing required parameters")
        print("\nℹ️  Usage examples:")
        print("  1. Compare with loaded encodings:")
        print("     python final.py --load face_recognition.pkl --test test.jpg")
        print("  2. Train and save:")
        print("     python final.py --mode train --known ./known_faces/ --save encodings.pkl")
        print("  3. Real-time recognition:")
        print("     python final.py --mode realtime --load encodings.pkl")
        print("  4. One-to-one comparison:")
        print("     python final.py --known person1.jpg --test test.jpg")
        print("\n🔧 Options:")
        print("  --tolerance 0.5    # Stricter matching (default: 0.6)")
        print("  --mode compare/train/realtime/test")

def demo_simple_comparison():
    """Simple demonstration without command line arguments"""
    system = FaceRecognitionSystem(tolerance=0.6)

    print("🧪 Running face recognition demo...")
    print("=" * 50)

    # Example 1: One-to-one comparison (modify paths as needed)
    print("\n1️⃣ One-to-One Comparison:")
    result = system.test_single_comparison(
        "data/stilefile.jpg",
        "data/djoms.jpeg"
    )

    print(f"\n📊 Result:")
    print(f"   Match: {result.get('match', 'N/A')}")
    print(f"   Distance: {result.get('distance', 'N/A'):.4f}")
    print(f"   Confidence: {result.get('confidence', 'N/A'):.1f}%")

    # Example 2: Load multiple known faces (if directory exists)
    known_faces_dir = "known_faces"
    if os.path.exists(known_faces_dir) and os.path.isdir(known_faces_dir):
        print(f"\n2️⃣ Database Mode:")
        system.load_known_faces_from_directory(known_faces_dir)
        system.print_database_info()

        # Test with a known image
        test_image = "test_face.jpg"
        if os.path.exists(test_image):
            print(f"\n3️⃣ Database Comparison:")
            results = system.compare_faces(test_image)

    print("\n✅ Demo completed!")


if __name__ == "__main__":
    # If no command line arguments, run demo
    import sys
    if len(sys.argv) == 1:
        demo_simple_comparison()
    else:
        main()