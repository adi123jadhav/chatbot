window.onload = function () {
    appendMessage("BOT", "Hi! How can I help you today?");
};
document.getElementById("user-input").addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});

function appendMessage(sender, message) {
    const chatBox = document.getElementById("chat-container");
    const messageElement = document.createElement("div");
    messageElement.classList.add("chat-message", sender === "You" ? "user" : "bot");

    // For bot messages, handle HTML properly for line breaks
    if (sender === "BOT") {
        messageElement.innerHTML = message.replace(/\n/g, "<br>"); // Convert newlines to <br>
    } else {
        messageElement.textContent = message; // Plain text for user input
    }

    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to the latest message
}

function startRecording() {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const recognition = new (window.webkitSpeechRecognition || window.SpeechRecognition)();
        recognition.lang = 'en-US'; // Set language
        recognition.interimResults = true; // Show interim results for feedback

        const popup = document.getElementById('recording-popup');
        const transcriptDisplay = document.getElementById('transcript-display');

        recognition.onstart = () => {
            popup.classList.remove('hidden'); // Show the pop-up
            transcriptDisplay.textContent = "Listening..."; // Reset transcript display
        };

        recognition.onresult = (event) => {
            const interimTranscript = Array.from(event.results)
                .map(result => result[0].transcript)
                .join('');
            transcriptDisplay.textContent = `You said: ${interimTranscript}`;
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            alert('Error: ' + event.error);
        };

        recognition.onend = () => {
            popup.classList.add('hidden'); // Hide the pop-up
            const finalTranscript = transcriptDisplay.textContent.replace("You said: ", "");
            document.getElementById('user-input').value = finalTranscript; // Populate text box
            sendMessage(); // Automatically send the message
        };

        recognition.start();
    } else {
        alert("Your browser does not support voice input.");
    }
}


function sendMessage() {
    const userInput = document.getElementById("user-input").value.trim();
    if (!userInput) return;

    appendMessage("You", userInput);
    document.getElementById("user-input").value = "";

    appendMessage("BOT", "Typing...");

    fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: userInput })
    })
        .then(response => response.json())
        .then(data => {
            const chatBox = document.getElementById("chat-container");
            const typingIndicator = chatBox.querySelector(".bot:last-child");
            if (typingIndicator && typingIndicator.textContent === "Typing...") {
                typingIndicator.remove();
            }
            appendMessage("BOT", data.response);
        })
        .catch(error => {
            console.error("Error:", error);
            appendMessage("BOT", "Sorry, there was an error processing your request.");
        });
}
