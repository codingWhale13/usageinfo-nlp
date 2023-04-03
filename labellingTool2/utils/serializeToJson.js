export function dateToString(object) {
  function convert(object, k) {
    if (object[k] instanceof Date) {
      return (object[k] = object[k].toString());
    }
  }
  eachRecursive(object, convert);
  eachRecursive(object, undefinedToNull);
  return object;
}

export function undefinedToNull(object, key) {
  if (object[key] === undefined) {
    object[key] = null;
  }
}
function eachRecursive(obj, f) {
  for (var k in obj) {
    f(obj, k);
    if (typeof obj[k] == "object" && obj[k] !== null) {
      eachRecursive(obj[k], f);
    }
  }
}
