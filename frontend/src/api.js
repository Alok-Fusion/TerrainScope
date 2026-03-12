const RAW_API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export const API_BASE = RAW_API_BASE.replace(/\/$/, "");

export function resolveAssetUrl(path) {
  if (!path) {
    return "";
  }
  if (path.startsWith("data:") || path.startsWith("blob:") || path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  return `${API_BASE}${path}`;
}

export async function apiFetch(path, options = {}) {
  const response = await fetch(resolveAssetUrl(path), options);
  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch (error) {
      const text = await response.text();
      message = text || message;
    }
    throw new Error(message);
  }
  return response.json();
}
