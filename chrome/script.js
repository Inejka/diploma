// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// A generic onclick callback function.
chrome.contextMenus.onClicked.addListener(genericOnClick);


// A generic onclick callback function.
async function genericOnClick(info, tab) {
    const item = await chrome.storage.sync.get(['latest_url']);
    fetch(item.latest_url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json; charset=UTF-8'
        },
        body: JSON.stringify({
            data: [info.srcUrl, info.menuItemId],
            event_data: null,
            fn_index: 0
        })
    })
        .then(response => response)
        .then(data => {
            console.log('Success:', data);
        })
        .catch((error) => {
            console.error('Error:', error);
        });


}

chrome.runtime.onInstalled.addListener(function () {
    // Create one test item for each context type.

    let id = chrome.contextMenus.create({
        title: 'Score image',
        contexts: ['image'],
        id: 'image'
    })

    for (let i = 1; i < 11; i++) {
        let id1 = chrome.contextMenus.create({
            title: i.toString(),
            parentId: id,
            contexts: ['image'],
            id: i.toString()
        })
    }

});


