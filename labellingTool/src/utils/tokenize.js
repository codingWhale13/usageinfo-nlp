function tokenizeString(string) {
  return string
    .replaceAll('<br />', '') // remove new lines (\n doesn't work, sadly...)
    .replaceAll('\\', '') // these guys are sometimes in the text
    .replaceAll('&#34;', '"') // show quotes
    .replaceAll(/[^\w\s-']|_/g, function ($1) {
      return ' ' + $1 + ' ';
    })
    .replaceAll(/[ ]+/g, ' ') // replace multiple spaces with single space
    .split(' ');
}

module.exports = tokenizeString;
