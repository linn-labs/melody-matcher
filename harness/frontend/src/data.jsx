// Static lookup tables only. Models and runs come from the backend.

const GENRES = [
  "Indie", "Ambient", "Electronic", "R&B", "Rock", "Pop",
  "Folk", "Jazz", "Hip-Hop", "Country", "Classical", "Metal",
];

// Deterministic hue from a string — stable artwork tile color per track.
function hueFor(str) {
  if (!str) return 20;
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = (h * 31 + str.charCodeAt(i)) >>> 0;
  }
  return h % 360;
}

Object.assign(window, { GENRES, hueFor });
