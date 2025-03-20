document.addEventListener("DOMContentLoaded", async function () {
    const generateButton = document.getElementById("generate-button");
    const resultText = document.getElementById("generated-tweet");
    const loadingSpinner = document.getElementById("loading-spinner");
    const loadingMessage = document.getElementById("loading-message");
    // Load ONNX model
    loadingSpinner.style.display = "flex";  // Show spinner
    let session;
    try {
        session = await ort.InferenceSession.create("./model.onnx");
        console.log("ONNX model loaded successfully.");
        loadingSpinner.style.display = "none";
    } catch (error) {
        console.error("Error loading ONNX model:", error);
        loadingMessage.textContent = "Load failed. Please refresh.";
        return;
    }
    // Character-to-index dictionary for encoding input text
    let intToChar = {0: '\n', 1: ' ', 2: '!', 3: '"', 4: '#', 5: '$', 6: '%', 7: '&', 8: "'", 9: '(', 10: ')', 11: '*', 12: '+', 13: ',', 14: '-', 15: '.', 16: '/', 17: '0', 18: '1', 19: '2', 20: '3', 21: '4', 22: '5', 23: '6', 24: '7', 25: '8', 26: '9', 27: ':', 28: ';', 29: '=', 30: '?', 31: '@', 32: 'A', 33: 'B', 34: 'C', 35: 'D', 36: 'E', 37: 'F', 38: 'G', 39: 'H', 40: 'I', 41: 'J', 42: 'K', 43: 'L', 44: 'M', 45: 'N', 46: 'O', 47: 'P', 48: 'Q', 49: 'R', 50: 'S', 51: 'T', 52: 'U', 53: 'V', 54: 'W', 55: 'X', 56: 'Y', 57: 'Z', 58: '[', 59: ']', 60: '_', 61: 'a', 62: 'b', 63: 'c', 64: 'd', 65: 'e', 66: 'f', 67: 'g', 68: 'h', 69: 'i', 70: 'j', 71: 'k', 72: 'l', 73: 'm', 74: 'n', 75: 'o', 76: 'p', 77: 'q', 78: 'r', 79: 's', 80: 't', 81: 'u', 82: 'v', 83: 'w', 84: 'x', 85: 'y', 86: 'z', 87: '{', 88: '|', 89: '}', 90: '~', 91: 'à', 92: 'á', 93: 'è', 94: 'é', 95: 'ë', 96: 'ñ', 97: 'ó', 98: 'ú', 99: 'ʉ', 100: '̱', 101: 'ω', 102: 'я', 103: 'ӕ', 104: 'ԍ', 105: 'ԏ', 106: 'Ԡ', 107: 'ե', 108: 'լ', 109: 'ջ', 110: 'ُ', 111: '٪', 112: '\u06dd', 113: 'ۢ', 114: '۪', 115: '\u2005', 116: '–', 117: '—', 118: '‘', 119: '’', 120: '“', 121: '”', 122: '…',
        123: '\u205f'};

    let abortController = null; // Store an abort controller for cancellatio

    generateButton.addEventListener("click", async function () {
        if (abortController) {
            abortController.abort(); // Cancel previous generation if it's running
        }
        abortController = new AbortController(); // Create a new one for the current run
        const { signal } = abortController;

        const newChars = 280;  // Number of characters to generate (as in the Python code)
        const minChars = 180;  // Allow shorter tweets for variation
        const vocab_size = 124;
        const context_length = 128;  // Set a context length similar to the model's configuration

        // Initialize the context with zeros (similar to the Python code where context is a tensor of zeros)
        let context = [[0]];

        resultText.textContent = "";  // Clear previous text
        const delay = 30;
        let nextChar = "";
        let sentenceEnded = false;

        // Run the generation loop
        for (let i = 0; i < newChars; i++) {
            if (signal.aborted) return; // Stop immediately if canceled
            // Ensure the context length doesn't exceed the maximum allowed length
            if (context[0].length > context_length) {
                context[0] = context[0].slice(-context_length);
            }

            // Create a tensor for the current context
            let contextTensor = new ort.Tensor("int32", new Int32Array(context[0]), [1, context[0].length]);

            try {
                // Run the model to predict the next character
                const outputMap = await session.run({ input: contextTensor });
                const outputTensor = outputMap.output;
                
                // Get the predictions for the next character (last time step)
                const prediction = outputTensor.data.slice(-vocab_size); // The last element (next token)

                // Apply softmax to get the probabilities for the next character
                const probabilities = softmax(prediction);

                // Sample the next character based on the probabilities
                const nextCharIndex = sample(probabilities);
                nextChar = intToChar[nextCharIndex];

                // Append the predicted character to the result
                resultText.textContent += nextChar;

                // Update the context with the new character
                context[0].push(nextCharIndex);

                await new Promise(resolve => setTimeout(resolve, delay));

                // Check if the last character is a sentence-ending punctuation
                if (i >= minChars && (nextChar === "." || nextChar === "!" || nextChar === "?")) {
                    sentenceEnded = true;
                }

                // If a sentence has ended and there are at least minChars, break
                if (sentenceEnded && i >= minChars) break;
            } catch (error) {
                console.error("Error during inference:", error);
                break; // If there's an error, stop the generation loop
            }
        }
    });

    // Softmax function to calculate probabilities
    function softmax(arr) {
        const maxVal = Math.max(...arr);
        const expArr = arr.map(val => Math.exp(val - maxVal));
        const sumExp = expArr.reduce((acc, val) => acc + val, 0);
        return expArr.map(val => val / sumExp);
      }

    // Sampling function to select a character based on probabilities
    function sample(probabilities) {
        let rand = Math.random();
        let cumulativeProbability = 0;

        for (let i = 0; i < probabilities.length; i++) {
            cumulativeProbability += probabilities[i];
            if (rand < cumulativeProbability) {
                return i;
            }
        }
        return probabilities.length - 1;  // Default case (shouldn't normally happen)
    }
});