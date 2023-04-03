export function annotationsToUsageOptions(annotations) {
  return annotations.map((annotation) => annotation.tokens.join(" ")).flat();
}

export function uniqueAnnotations(annotations) {
  return annotations.filter(
    (annotation, index) =>
      annotations
        .map((annotation) => annotation.tokens.join(" "))
        .indexOf(annotation.tokens.join(" ")) === index
  );
}
