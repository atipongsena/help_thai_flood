// Monotone Chain Convex Hull Algorithm
// Returns an array of [lat, lng] coordinates representing the convex hull polygon

type Point = [number, number]; // [lat, lng]

function crossProduct(o: Point, a: Point, b: Point): number {
  return (a[1] - o[1]) * (b[0] - o[0]) - (a[0] - o[0]) * (b[1] - o[1]);
}

export function getConvexHull(points: Point[]): Point[] {
  const n = points.length;
  if (n <= 3) return points;

  // Sort points by lat, then lng
  const sortedPoints = [...points].sort((a, b) => {
    return a[0] === b[0] ? a[1] - b[1] : a[0] - b[0];
  });

  const lower: Point[] = [];
  for (const p of sortedPoints) {
    while (lower.length >= 2 && crossProduct(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }

  const upper: Point[] = [];
  for (let i = n - 1; i >= 0; i--) {
    const p = sortedPoints[i];
    while (upper.length >= 2 && crossProduct(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }

  upper.pop();
  lower.pop();
  return lower.concat(upper);
}
