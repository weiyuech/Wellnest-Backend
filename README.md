# 🚀 Wellnest Backend – FastAPI Deployment on Render

This is the backend API for **Wellnest**, a pregnancy risk prediction application powered by FastAPI.

---

## 📦 Tech Stack

- **FastAPI** – Modern, fast (high-performance) web framework
- **Uvicorn** – ASGI server
- **Python 3.10+**

---

## 🌐 Live Deployment (via [Render.com](https://render.com))

This project is deployed using **Render's native Python environment** (not Docker).

### ✅ Deployment Steps (Render)

1. **Create a GitHub repository** (if you haven't already)
2. **Push this project** to your GitHub repo
3. Go to [https://dashboard.render.com](https://dashboard.render.com) and:
   - Click **"New Web Service"**
   - Connect your GitHub repo
   - Configure the settings as follows:

| Setting         | Value                                                        |
|----------------|--------------------------------------------------------------|
| **Language**    | Python 3                                                     |
| **Build Command** | `pip install -r requirements.txt`                          |
| **Start Command** | `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT` |

> 🔁 Adjust the `Start Command` if your FastAPI app is located in a different file.

---

