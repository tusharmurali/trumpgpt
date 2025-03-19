// document.addEventListener("DOMContentLoaded", async function () {
//     const generateButton = document.getElementById("generate-button");
//     const resultText = document.getElementById("generated-tweet");
//     const loadingMessage = document.getElementById("loading-message");

//     // Load the ONNX model
//     let session;
//     try {
//         session = await ort.InferenceSession.create("./model.onnx");
//         console.log("ONNX model loaded successfully.");
//     } catch (error) {
//         console.error("Error loading ONNX model:", error);
//         loadingMessage.textContent = "Failed to load AI model.";
//         return;
//     }

//     generateButton.addEventListener("click", async function () {
//         const inputText = document.getElementById("prompt-input").value.trim();
//         if (!session) return;

//         loadingMessage.textContent = "Generating tweet...";
//         resultText.textContent = "";

//         // Convert input to tensor
//         const inputTensor = new ort.Tensor("int32", preprocessInput(inputText), [1, inputText.length]);
        
//         try {
//             // Run inference
//             const outputs = await session.run({ "input": inputTensor });
//             const outputArray = outputs["output"].data;
//             const generatedText = postprocessOutput(outputArray);
            
//             resultText.textContent = generatedText;
//         } catch (error) {
//             console.error("Error generating text:", error);
//             resultText.textContent = "Error generating tweet.";
//         }

//         loadingMessage.textContent = "";
//     });

//     function preprocessInput(text) {
//         // Convert text to tokenized numerical input (this should match training preprocessing)
//         return text.split("").map(char => charToInt[char] || 0);
//     }

//     function postprocessOutput(outputArray) {
//         // Convert tokenized output back to string
//         return outputArray.map(idx => intToChar[idx] || "").join("");
//     }
// });


document.addEventListener("DOMContentLoaded", async function () {
    const generateButton = document.getElementById("generate-button");
    const resultText = document.getElementById("generated-tweet");
    // Load ONNX model
    let session;
    try {
        session = await ort.InferenceSession.create("./model.onnx");
        console.log("ONNX model loaded successfully.");
    } catch (error) {
        console.error("Error loading ONNX model:", error);
        return;
    }

    // Character-to-index dictionary for encoding input text
    let intToChar = {0: '\n', 1: ' ', 2: '!', 3: '"', 4: '$', 5: '%', 6: '&', 7: "'", 8: '(', 9: ')', 10: '*', 11: '+', 12: ',', 13: '-', 14: '.', 15: '/', 16: '0', 17: '1', 18: '2', 19: '3', 20: '4', 21: '5', 22: '6', 23: '7', 24: '8', 25: '9', 26: ':', 27: ';', 28: '?', 29: 'A', 30: 'B', 31: 'C', 32: 'D', 33: 'E', 34: 'F', 35: 'G', 36: 'H', 37: 'I', 38: 'J', 39: 'K', 40: 'L', 41: 'M', 42: 'N', 43: 'O', 44: 'P', 45: 'Q', 46: 'R', 47: 'S', 48: 'T', 49: 'U', 50: 'V', 51: 'W', 52: 'X', 53: 'Y', 54: 'Z', 55: '[', 56: ']', 57: '_', 58: 'a', 59: 'b', 60: 'c', 61: 'd', 62: 'e', 63: 'f', 64: 'g', 65: 'h', 66: 'i', 67: 'j', 68: 'k', 69: 'l', 70: 'm', 71: 'n', 72: 'o', 73: 'p', 74: 'q', 75: 'r', 76: 's', 77: 't', 78: 'u', 79: 'v', 80: 'w', 81: 'x', 82: 'y', 83: 'z', 84: '{', 85: '|', 86: '}', 87: 'à', 88: 'á', 89: 'è', 90: 'é', 91: 'ë', 92: 'ñ', 93: 'ó', 94: 'ú', 95: '\u2005', 96: '–', 97: '—', 98: '‘', 99: '’', 100: '“', 101: '”', 102: '…', 103: '\u205f'};

    generateButton.addEventListener("click", async function () {
        const newChars = 280;  // Number of characters to generate (as in the Python code)
        const vocab_size = 104;
        const context_length = 128;  // Set a context length similar to the model's configuration

        // Initialize the context with zeros (similar to the Python code where context is a tensor of zeros)
        let context = [[0, 41]];

        // Start with an empty generated text
        // let generatedText = "";

        resultText.textContent = "";  // Clear previous text

        // Run the generation loop
        for (let i = 0; i < newChars; i++) {
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
                // let last_time_step = prediction[prediction.length - 1];
                // if (context[0].length == 1) last_time_step = prediction

                // Apply softmax to get the probabilities for the next character
                const probabilities = softmax(prediction);

                // Sample the next character based on the probabilities
                const nextCharIndex = sample(probabilities);

                // Append the predicted character to the result
                // generatedText += intToChar[nextCharIndex];
                resultText.textContent += intToChar[nextCharIndex]

                // Update the context with the new character
                context[0].push(nextCharIndex);
            } catch (error) {
                console.error("Error during inference:", error);
                break; // If there's an error, stop the generation loop
            }
        }

        // Display the generated tweet
        // resultText.textContent = generatedText;
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