import cv2
import numpy as np
import face_recognition
import os
from pathlib import Path

def load_and_convert_image(image_path):
    """Load and convert image with error handling"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None

    try:
        image = face_recognition.load_image_file(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def detect_and_encode_faces(image):
    """Detect faces and return encodings with locations"""
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if len(face_encodings) == 0:
        print("No faces detected in the image!")
        return None, None

    return face_locations, face_encodings

def draw_face_rectangle(image, face_location, color=(255, 0, 0), thickness=2):
    """Draw rectangle around detected face"""
    top, right, bottom, left = face_location
    cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
    return image

def main():
    # File paths
    known_image_path = 'data/stilefile.jpg'
    test_image_path = 'data/djoms.jpeg'

    # Load images
    known_image = load_and_convert_image(known_image_path)
    test_image = load_and_convert_image(test_image_path)

    if known_image is None or test_image is None:
        return

    # Display original images
    cv2.imshow('Known Person', known_image)
    cv2.imshow('Test Person', test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Detect and encode faces in known image
    known_face_locations, known_face_encodings = detect_and_encode_faces(known_image)
    if known_face_encodings is None:
        return

    # Detect and encode faces in test image
    test_face_locations, test_face_encodings = detect_and_encode_faces(test_image)
    if test_face_encodings is None:
        return

    # Draw rectangles on known image
    for i, location in enumerate(known_face_locations):
        known_image = draw_face_rectangle(known_image.copy(), location)
        cv2.imshow(f'Known Person - Face {i+1}', known_image)
        cv2.waitKey(0)

    # Compare faces
    results = []
    face_distances = []

    for test_encoding in test_face_encodings:
        # Compare with each known face encoding
        for known_encoding in known_face_encodings:
            result = face_recognition.compare_faces([known_encoding], test_encoding)
            distance = face_recognition.face_distance([known_encoding], test_encoding)
            results.append(result[0])
            face_distances.append(round(distance[0], 4))

    # Display results
    for i, (result, distance) in enumerate(zip(results, face_distances)):
        # Draw on test image
        test_display = test_image.copy()

        # Draw rectangle around detected face
        if i < len(test_face_locations):
            test_display = draw_face_rectangle(test_display, test_face_locations[i], color=(0, 255, 0))

        # Add text with results
        status = "MATCH" if result else "NO MATCH"
        color = (0, 255, 0) if result else (0, 0, 255)

        cv2.putText(test_display,
                    f'{status} - Distance: {distance}',
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color, 2)

        cv2.imshow(f'Comparison Result {i+1}', test_display)
        cv2.waitKey(0)

    print("\n=== FACE RECOGNITION RESULTS ===")
    print(f"Known faces detected: {len(known_face_encodings)}")
    print(f"Test faces detected: {len(test_face_encodings)}")

    for i, (result, distance) in enumerate(zip(results, face_distances)):
        print(f"\nComparison {i+1}:")
        print(f"  Match: {result}")
        print(f"  Distance: {distance}")
        print(f"  Confidence: {(1 - distance) * 100:.2f}%")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()