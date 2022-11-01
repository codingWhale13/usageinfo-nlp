function tokenizeString(string){
    return string.replace(/[^\w\s]|_/g, function ($1) { return ' ' + $1 + ' ';}).replace(/[ ]+/g, ' ').split(' ');
}

module.exports = tokenizeString;