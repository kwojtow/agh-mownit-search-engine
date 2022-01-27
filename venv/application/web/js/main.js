async function run() {
    let ids = ["normal", "normalized", "reduced"]
    let headers = ["Matrix withoud IDF", "Normalized matrix", "Reduced matrix"]
    for (var i = 0; i < 3; i++){
        document.getElementById(ids[i]).innerHTML = ''
    }
    x = document.getElementById('input').value;
    let n = await eel.get_results(x)();

    for (var i = 0; i < 3; i++){
        let HTML = '<h2>' + headers[i] + ' results:<h2>'
        console.log(n[i])
        var tmp = n[i]
        console.log(tmp[0])
        for(var j = 0; j < tmp.length; j++){
            let el = tmp[j]
            HTML += "<h3><a href=\"" + el[0] + "\">" + el[1] + "</a></h3><span>Match level: " + el[2] + " </span>"
        }
        document.getElementById(ids[i]).innerHTML = HTML
    }
}
document.getElementById('search').addEventListener('click', run)
document.getElementById('input').addEventListener('keypress', function(e){
    if(e.key === 'Enter'){
        run()
    }
})