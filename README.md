# Milvus Image Embedding Demo

This project demonstrates how to extract image embeddings using a pre-trained ResNet50 model and store them in a Milvus vector database. The stack runs locally using Docker Compose and provides a UI for managing collections via Attu.

## Project Structure

- `main.py` — Extracts embeddings from images and inserts them into Milvus.
- `docker-compose.yml` — Sets up Milvus, MinIO, Etcd, and Attu (Milvus UI).

## Getting Started

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Start Milvus and Dependencies

```sh
docker-compose up -d
```

Wait until all services are healthy (especially `milvus-standalone` and `milvus-attu`).

### 3. Install Python Requirements

It's recommended to use a virtual environment:

```sh
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision pymilvus pillow
```

### 4. Insert an Image Embedding

Run the script with the path to your image:

```sh
python main.py <image_path>
```

Example:

```sh
python main.py ./my_image.jpg
```

### 5. Access Milvus UI

Open your browser and go to: [http://localhost:3000](http://localhost:3000)

- Default Milvus connection: `milvus-standalone:19530`
- Use the UI to explore collections and data.

## Notes

- Embeddings are stored in a collection named `images` with fields: `id`, `embedding`, and `image_path`.
- Data persists in the `volumes/` directory.
- Default MinIO credentials: `minioadmin` / `minioadmin` (not required for basic usage).

## Troubleshooting

- If you see connection errors, ensure all Docker containers are running and healthy.
- For Milvus or Attu issues, check logs with `docker-compose logs <service-name>`.

---

Enjoy experimenting
