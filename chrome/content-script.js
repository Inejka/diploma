// Define a minimum image size
const MIN_IMAGE_SIZE = 100;

// Select all images on the page

document.sended_images = new Map()

async function addRandomNumberToImage(image) {
    if (image.width < MIN_IMAGE_SIZE || image.height < MIN_IMAGE_SIZE) {
        return;
    }

    if (document.sended_images.has(image["src"])) {
        return;
    }
    document.sended_images.set(image["src"], "TEMP");


    // Create a new div element for the random number
    const number = document.createElement('div');

    // Calculate the size of the number square based on the image size
    const imageSize = Math.min(image.width, image.height);
    const numberSize = imageSize / 6;
    console.log("Will send" + image['src'])

    var response = await fetch(document.url_with_model, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json; charset=UTF-8'
        },
        body: JSON.stringify({
            data: [image['src']],
            event_data: null,
            fn_index: 2
        })
    })

    var not_promise = await response.json()

    // Generate a random number between 1 and 100
    const randomNumber = not_promise["data"][0];
    console.log("Get ansered with " + randomNumber.toString());
    document.sended_images.set(image["src"], randomNumber);

    // Set the position and size of the number square
    number.style.position = 'absolute';
    number.style.top = `${image.offsetTop + 20}px`;
    number.style.left = `${image.offsetLeft + (image.width - numberSize) - 20}px`;
    number.style.width = `${numberSize}px`;
    number.style.height = `${numberSize}px`;

    // Add the random number
    number.textContent = randomNumber;
    number.style.fontSize = `${numberSize / 2}px`;
    number.style.fontWeight = 'bold';
    number.style.color = 'white';
    number.style.backgroundColor = 'red';
    number.style.textAlign = 'center';
    number.style.lineHeight = `${numberSize}px`;
    number.style.borderRadius = `${numberSize / 4}px`;

    // Append the number to the parent of the image
    image.parentNode.appendChild(number);

    // Check if the image is too small

}

async function scoreImages() {
    var url_promise = await chrome.storage.sync.get(['latest_url'])
    var url = await url_promise.latest_url
    document.url_with_model = url
    const images = document.querySelectorAll('img');

    // Loop through each image and add a random number
    for (const item of images) {
        await addRandomNumberToImage(item);
    }
}

setInterval(scoreImages, 1000)