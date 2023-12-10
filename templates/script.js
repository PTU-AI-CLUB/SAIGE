
document.getElementById('send-button').addEventListener('click', sendMessage);
document.getElementById('message-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function sendMessage() {
    const textField = document.getElementById('message-input');
    const text = textField.value.trim();
    const loadingDiv = document.getElementById('loading');

    if (text === '') return;

    displayMessage("user-message", "You", text);
    textField.value = '';
    loadingDiv.classList.remove('hidden');
    scrollToBottom();

    fetch('http://129.154.38.226:5000/saige', {
        method: 'POST',
        body: JSON.stringify({ message: text }),
        mode: 'cors',
        headers: {
            'Content-Type': 'application/json'
        },
    })
    .then(response => response.json())
    .then(data => {
        loadingDiv.classList.add('hidden');
        displayMessage("bot-message", "Saige", data.answer);
        scrollToBottom();
    }).catch(error => {
        console.error('Error:', error);
        loadingDiv.classList.add('hidden');
    });
}

function displayMessage(className, sender, message) {
    const messagesDiv = document.getElementById('messages');
    messagesDiv.innerHTML += `<div class="${className} message"><b>${sender}:</b> ${message}</div>`;
}

function scrollToBottom() {
    const messagesDiv = document.getElementById('messages');
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}
