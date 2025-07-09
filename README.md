
# 🧱 3D Mesh Reconstruction Pipeline from Gaussian Splatting Point Clouds

This project presents a **Python-based pipeline** for converting 3D Gaussian Splatting point clouds into **high-quality textured 3D meshes**, ideal for indie game developers and animators. While the Gaussian Splatting model itself is **not included**, this repository provides a detailed step-by-step guide for generating the point cloud and processing it into optimized meshes.

---

## 🧠 Core Idea

- 📷 Start with image captures
- ➡️ Process through COLMAP + Gaussian Splatting to generate a point cloud
- 🧵 Run this repo's code to convert the point cloud to a mesh using:
  - **Poisson Surface Reconstruction**
  - **Ball Pivoting Algorithm (BPA)**
  - **Alpha Shapes**
- 🚀 GPU-accelerated densification and SH-to-RGB color mapping
- 🎨 Final output: a fully colored 3D mesh suitable for Blender, Unity, or Unreal

---

## 📸 Step 1: Image Capture & Preprocessing

Take multiple high-quality images of your object or scene from various angles (preferably full 360°). Ensure:
- Consistent lighting
- Sharp focus
- Minimal background clutter

### 🔧 Optional: Enhance Image Quality

Use the included script to batch-enhance images (brightness, contrast, noise reduction):

```bash
python image_enchacement.py
```

Edit `input_folder` and `output_folder` at the bottom of the script to match your dataset.

---

## 🧭 Step 2: Structure from Motion using COLMAP

1. Install [COLMAP](https://colmap.github.io/).
2. Open COLMAP GUI or run via CLI to:
   - Create a new project
   - Import your images
   - Run Feature Extraction → Matching → Sparse Reconstruction (Structure-from-Motion)
   - Run Dense Reconstruction (Multi-View Stereo)

3. Export your COLMAP model:
```
<your_colmap_workspace>/sparse/
<your_colmap_workspace>/images.txt
```

---

## 🌈 Step 3: Gaussian Splatting Point Cloud

1. Clone and install [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).
2. Convert COLMAP output to the format required by GS.
3. Train the Gaussian model:
```bash
python train.py -s <your_scene_folder>
```
4. After training, export the point cloud:
```
<your_scene_folder>/output/point_cloud/point_cloud.ply
```

This file is the input to our reconstruction pipeline.

---

## 🧵 Step 4: Mesh Reconstruction Pipeline (This Repo)

### 🔧 Requirements

```bash
pip install open3d cupy scipy plyfile pillow tqdm
```

> ⚠️ CuPy requires a CUDA-compatible GPU.

---

### ▶️ How to Run

1. Open `Ply_to_Mesh.py`  
2. Replace the default point cloud path with yours:
```python
path = "path/to/your/point_cloud.ply"
```

3. Run the script:
```bash
python Ply_to_Mesh.py
```

This will generate and visualize the final 3D mesh.

---

## 🔍 Pipeline Steps (Under the Hood)

✔️ Convert Spherical Harmonic (SH) colors → RGB  
✔️ Load & visualize initial point cloud  
✔️ **Densify the point cloud** using GPU with CuPy  
✔️ Add volume using Poisson disk sampling  
✔️ Clean and filter the point cloud  
✔️ Estimate normals for all meshing methods  
✔️ Generate meshes:
- **Poisson Surface Reconstruction**
- **Ball Pivoting Algorithm (BPA)**
- **Alpha Shape Meshing**

✔️ Assign colors to all mesh vertices  
✔️ Clean, simplify, and smooth the meshes  
✔️ Combine all meshes into one final `.ply`

Final output: `output_final_mesh_colored.ply`

---

## 🗂️ Repo Structure

```
📁 input_images/                   # Raw images (optional)
📁 output_images/                  # Enhanced images (optional)
📄 image_enchacement.py            # Script to enhance image quality
📄 Ply_to_Mesh.py                  # Main reconstruction pipeline
📄 output_final_mesh_colored.ply  # Final generated mesh (after run)
```

---

## 🎯 Use Cases

- 🎮 Game Development (Unity, Unreal Engine)
- 🎬 Animation & VFX (Blender, Maya)
- 🎓 Research & 3D Scene Understanding
- 🧠 AI & Simulation

---

## 🌱 Future Work

- Improve mesh quality in sparse regions
- Reduce the number of input images needed for splatting
- Support for multi-texture maps and semantic segmentation
- Integration with real-time game engines

---

## 👨‍💻 Author

**Raj Mistry**  
MSc Data Science, University of Surrey  
📌 [Portfolio](www.linkedin.com/in/rajmistry16)

---

## 🙏 Acknowledgements

- [COLMAP](https://colmap.github.io/)
- [Gaussian Splatting by INRIAGraphDECO](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Open3D](http://www.open3d.org/)
