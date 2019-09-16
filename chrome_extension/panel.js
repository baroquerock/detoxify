function reveal() {
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    var tab = tabs[0];
    chrome.tabs.sendMessage(tab.id, {action: 'reveal_toxicity'}, function(response) {});
  });
}

function reset() {
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    var tab = tabs[0];
    chrome.tabs.sendMessage(tab.id, {action: 'reset'}, function(response) {});
  });
}


document.addEventListener('DOMContentLoaded', function() {

    document.querySelector('#toxicity_index').addEventListener(
        'click', reveal);

    document.querySelector('#reset_toxic').addEventListener(
        'click', reset);

});