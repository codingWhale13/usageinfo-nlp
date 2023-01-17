export function annotationsToUsageOptions(annotations){
    return annotations.map(annotation => annotation.tokens.join(' ')).flat();
};