import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Load the dictionary that was used to generate the markers.
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# Initialize the detector parameters using default values.
parameters = cv2.aruco.DetectorParameters_create()

# Initialize the graph.
graph = nx.Graph()

# Define the origin marker ID and its position.
origin_marker_id = 0
origin_position = np.array([0, 0])

# Start the video capture from the external webcam.
cap = cv2.VideoCapture(1)

def add_marker_to_graph(marker_id, position):
    """Add a marker to the graph."""
    if marker_id not in graph:
        graph.add_node(marker_id, pos=position)
    else:
        graph.nodes[marker_id]['pos'] = position

def add_edge_to_graph(id1, id2, distance):
    """Add an edge between two markers in the graph."""
    if not graph.has_edge(id1, id2):
        graph.add_edge(id1, id2, weight=distance)

def calculate_relative_position(corner, reference_corner, reference_position):
    """Calculate the position of the marker relative to the reference marker."""
    marker_center = np.mean(corner, axis=0)
    reference_center = np.mean(reference_corner, axis=0)
    relative_position = marker_center - reference_center
    return reference_position + relative_position

while True:
    # Capture frame-by-frame.
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the markers in the image.
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Draw the detected markers.
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Initialize the dictionary to store the marker positions.
        marker_positions = {}

        # Loop through all the detected markers and update their positions in the graph.
        for marker_corner, marker_id in zip(corners, ids):
            marker_id = marker_id[0]

            if marker_id == origin_marker_id:
                marker_positions[marker_id] = origin_position
            elif marker_id not in graph:
                # If the marker is not in the graph, use the origin marker to calculate its position.
                marker_positions[marker_id] = calculate_relative_position(marker_corner[0], corners[0][0], origin_position)
            else:
                # If the marker is already in the graph, update its position based on the reference marker.
                marker_positions[marker_id] = calculate_relative_position(marker_corner[0], corners[0][0], graph.nodes[origin_marker_id]['pos'])

            # Draw the center of the marker.
            center = np.mean(marker_corner[0], axis=0)
            cv2.circle(frame, tuple(center.astype(int)), 5, (0, 255, 0), -1)
            cv2.putText(frame, f'ID: {marker_id}', tuple(center.astype(int) - [0, 10]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Add the marker to the graph.
            add_marker_to_graph(marker_id, marker_positions[marker_id])

        # Compute distances between detected markers and add edges to the graph.
        marker_ids = list(marker_positions.keys())
        for i in range(len(marker_ids)):
            for j in range(i + 1, len(marker_ids)):
                id1 = marker_ids[i]
                id2 = marker_ids[j]
                pos1 = marker_positions[id1]
                pos2 = marker_positions[id2]
                distance = np.linalg.norm(pos1 - pos2)
                add_edge_to_graph(id1, id2, distance)

    # Display the resulting frame.
    cv2.imshow('ArUco Marker Detection', frame)

    # Exit the loop when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close the windows.
cap.release()
cv2.destroyAllWindows()

# Draw the graph using matplotlib.
pos = nx.get_node_attributes(graph, 'pos')
nx.draw(graph, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_color='black', edge_color='gray')
labels = nx.get_edge_attributes(graph, 'weight')
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
plt.show()
