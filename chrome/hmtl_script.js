document.getElementById("submit").addEventListener("click", onConnectButton);

async function onConnectButton() {

    var to_add = '/run/predict'
    var url = document.getElementById('url_to_connect').value
    console.log(url)
    const output = document.getElementById("output-text");
    output.textContent = "Can't catch me? But if you can, there error somewhere";


    const xhr = new XMLHttpRequest();
    xhr.open("POST", url + to_add)
    xhr.setRequestHeader("Content-Type", "application/json; charset=UTF-8");
    const body = JSON.stringify({
        data: [],
        event_data: null,
        fn_index: 1
    });
    xhr.onload = () => {
        console.log("Get answered with" + JSON.parse(xhr.responseText))
        if (JSON.parse(xhr.responseText)["data"] == 'connected') {
            output.textContent = "connected";
            chrome.storage.sync.set({latest_url: (url + to_add)}).then(() => {
                console.log("Value is set to " + url + to_add);
            });
        }
    };
    xhr.send(body);
}

document.addEventListener("DOMContentLoaded", function () {
    info();
}, false);

async function info() {
    const item = await chrome.storage.sync.get(['latest_url']);
    const output = document.getElementById("output-text");
    output.textContent = "Will try to send info to: " + item.latest_url;

}
