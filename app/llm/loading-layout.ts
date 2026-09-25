/** A fixed crowd with a separate cell for each portrait and its reflection. */
export function loadingLayout(aspects: readonly number[]) {
  const columns = Math.ceil(Math.sqrt(aspects.length * 2));
  const rows = Math.ceil(aspects.length / columns);
  const cellWidth = 90 / columns;
  const cellHeight = 78 / rows;
  return aspects.map((aspect, character) => {
    const row = Math.floor(character / columns);
    const column = character % columns;
    const rowCount = Math.min(columns, aspects.length - row * columns);
    const variation = ((character * 7) % 5) / 4;
    // Reserve height for the visible half-reflection and the 14% rise animation.
    const height = cellHeight / 1.9 * (0.88 + variation * 0.12);
    const width = Math.min(cellWidth * 0.7, height * aspect / (16 / 9));
    return {
      character,
      x: 50 + (column - (rowCount - 1) / 2) * cellWidth + (variation - 0.5),
      y: 4 + row * cellHeight + cellHeight / 1.9 + variation,
      width,
      depth: row === rows - 1 ? "near" : row === 0 ? "far" : "mid",
      facing: column >= rowCount / 2 ? "left" : "right",
    };
  });
}
