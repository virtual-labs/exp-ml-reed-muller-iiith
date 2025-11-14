// --- 1. Define Code and Problem Globals ---

const GENERATOR_RM12 = [
    [1, 1, 1, 1], // v0
    [0, 1, 0, 1], // v1
    [0, 0, 1, 1]  // v2
];

const MESSAGES_RM12 = [
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
];

const CODEWORDS_RM12 = MESSAGES_RM12.map(msg => {
    const c0 = msg[0], c1 = msg[1], c2 = msg[2];
    const codeword = [0, 0, 0, 0];
    for (let i = 0; i < 4; i++) {
        codeword[i] = (c0 * GENERATOR_RM12[0][i] +
                       c1 * GENERATOR_RM12[1][i] +
                       c2 * GENERATOR_RM12[2][i]) % 2;
    }
    return codeword;
});

let solution = {};

// --- 2. Main Execution ---

// Hook up buttons immediately
document.getElementById('submitPart1Button').addEventListener('click', checkPart1);
document.getElementById('submitPart2Button').addEventListener('click', checkPart2);

/**
 * This function is called when MathJax is confirmed to be ready.
 */
function onMathJaxReady() {
    console.log("MathJax is ready, running setup.");
    setupProblem();
}

/**
 * Polls until the main MathJax typesetting function exists.
 */
function waitForMathJax() {
    // Check for the main object and the function we actually need
    if (window.MathJax && window.MathJax.typesetPromise) {
        // It's ready! Call the setup function.
        onMathJaxReady();
    } else {
        // It's not ready, poll again.
        console.log("Polling for MathJax...");
        setTimeout(waitForMathJax, 100);
    }
}

// Start polling for MathJax immediately
waitForMathJax();


// --- 3. Helper Functions ---

/**
 * Helper function to typeset MathJax content
 */
function typesetMath() {
    if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise().catch((err) => {
             console.log('MathJax typesetPromise failed:', err);
        });
    } else {
        console.log('typesetMath() called, but MathJax is not available.');
    }
}

/**
 * Generates a random number from a Gaussian distribution.
 */
function gaussianRandom(mean, stdDev) {
    let u, v, s;
    do {
        u = Math.random() * 2 - 1;
        v = Math.random() * 2 - 1;
        s = u * u + v * v;
    } while (s >= 1 || s === 0);
    let mul = Math.sqrt(-2.0 * Math.log(s) / s);
    return mean + stdDev * (v * mul); // Return one of the two samples
}

/**
 * Performs a Fast Hadamard Transform (FHT) on a 4-element vector.
 */
function fht(y) {
    // Stage 1
    const a0 = y[0] + y[1];
    const a1 = y[0] - y[1];
    const a2 = y[2] + y[3];
    const a3 = y[2] - y[3];

    // Stage 2
    const z0 = a0 + a2;
    const z1 = a1 + a3;
    const z2 = a0 - a2;
    const z3 = a1 - a3;

    return [z0, z1, z2, z3];
}

/**
 * Rounds a number to one decimal place.
 */
function roundToOneDecimal(num) {
    return Math.round(num * 10) / 10;
}

// --- 4. Main Problem Setup ---

function resetUI() {
    // Hide Part 2
    document.getElementById('part2Question').style.display = 'none';
    document.getElementById('part2-divider').style.display = 'none';
    document.getElementById('submitPart2Button').style.display = 'none';

    // Show Part 1
    document.getElementById('submitPart1Button').style.display = 'inline-block';

    // Clear inputs
    document.querySelectorAll('input[type="number"]').forEach(input => input.value = '');
    
    // Clear feedback
    const feedbackEl = document.getElementById('observation');
    feedbackEl.innerHTML = '<p>Your feedback will appear here.</p>';
    feedbackEl.className = ''; // Reset class
}

function setupProblem() {
    resetUI();
    
    // 1. Pick a random codeword and message
    const randomIndex = Math.floor(Math.random() * 8);
    const binaryCodeword = CODEWORDS_RM12[randomIndex];
    
    // 2. Convert to bipolar (+1 for 0, -1 for 1)
    const bipolarCodeword = binaryCodeword.map(bit => (bit === 0 ? 1 : -1));

    // 3. Add Gaussian noise
    const stdDev = 0.5;
    const receivedVectorY = bipolarCodeword.map(bit => bit + gaussianRandom(0, stdDev));

    // 4. Compute the correct FHT
    const correctTransformZ = fht(receivedVectorY);

    // --- 5. NEW: True ML Decoding Logic ---
    
    // 5a. Find index k with max absolute value
    let maxZ = -1;
    let k_ml = -1;
    for (let i = 0; i < 4; i++) {
        if (Math.abs(correctTransformZ[i]) > maxZ) {
            maxZ = Math.abs(correctTransformZ[i]);
            k_ml = i;
        }
    }

    // 5b. Determine ML message from k_ml
    const z_k = correctTransformZ[k_ml];
    
    // m_hat_0 is from the sign of z_k (z_k <= 0 maps to m_hat_0 = 1)
    const m_hat_0 = (z_k > 0) ? 0 : 1; 

    // m_hat_1 and m_hat_2 are from the index k_ml
    let m_hat_1 = 0;
    let m_hat_2 = 0;
    // k=0 -> (0,0) - default
    if (k_ml === 1) {      // k=1 -> (1,0)
        m_hat_1 = 1;
    } else if (k_ml === 2) { // k=2 -> (0,1)
        m_hat_2 = 1;
    } else if (k_ml === 3) { // k=3 -> (1,1)
        m_hat_1 = 1;
        m_hat_2 = 1;
    }

    const mlMessage = [m_hat_0, m_hat_1, m_hat_2];

    // 6. Reconstruct the *true* ML codeword
    let mlCodeword = [0, 0, 0, 0];
    for (let i = 0; i < 4; i++) {
        mlCodeword[i] = (mlMessage[0] * GENERATOR_RM12[0][i] +
                         mlMessage[1] * GENERATOR_RM12[1][i] +
                         mlMessage[2] * GENERATOR_RM12[2][i]) % 2;
    }

    // 7. Store solutions
    solution = {
        binaryCodeword: binaryCodeword, // The *original* codeword sent
        mlMessage: mlMessage,         // The ML message estimate from z
        mlCodeword: mlCodeword,       // The ML codeword estimate from z
        correctTransformZ: correctTransformZ,
        roundedTransformZ: correctTransformZ.map(roundToOneDecimal),
        k_ml: k_ml, // Store this for feedback
        z_k: z_k    // Store this for feedback
    };

    // 8. Update the UI
    const problemTextEl = document.getElementById('problem-text-container');
    const vectorString = receivedVectorY.map(n => n.toFixed(3)).join(', ');
    
    problemTextEl.innerHTML = `
        An \\(RM(1, 2)\\) codeword was sent using bipolar signaling (\\(0 \\to +1, 1 \\to -1\\)) over a noisy channel.
        <br>
        The received real vector \\(\\mathbf{y}\\) is:
        <div class="vector-display">
            \\(\\mathbf{y} = (${vectorString})\\)
        </div>
    `;
    
    // 9. Render the new MathJax
    typesetMath();
}

// --- 5. Check Answer Logic ---

function checkPart1() {
    // 1. Get user inputs
    const userTransform = [
        parseFloat(document.getElementById('z0').value),
        parseFloat(document.getElementById('z1').value),
        parseFloat(document.getElementById('z2').value),
        parseFloat(document.getElementById('z3').value)
    ];

    // 2. Check for empty inputs
    if (userTransform.some(isNaN)) {
        alert("Please fill in all 4 boxes for Part 1.");
        return;
    }
    
    // 3. Compare answers
    let transformCorrect = true;
    for (let i = 0; i < 4; i++) {
        // Use a small tolerance for floating point comparison
        if (Math.abs(userTransform[i] - solution.roundedTransformZ[i]) > 0.01) {
            transformCorrect = false;
            break;
        }
    }
    
    // 4. Generate feedback
    const feedbackEl = document.getElementById('observation');
    if (transformCorrect) {
        feedbackEl.className = 'obs-correct';
        feedbackEl.innerHTML = `
            <h4>Part 1 Correct!</h4>
            <p>Your transform values \(\\mathbf{z} = (${userTransform.join(', ')})\) are correct. Please proceed to Part 2.</p>
        `;
        
        // Show Part 2
        document.getElementById('part2Question').style.display = 'block';
        document.getElementById('part2-divider').style.display = 'block';
        document.getElementById('submitPart2Button').style.display = 'inline-block';
        document.getElementById('submitPart1Button').style.display = 'none';

    } else {
        feedbackEl.className = 'obs-incorrect';
        feedbackEl.innerHTML = `
            <h4>Part 1 Incorrect.</h4>
            <p>Your transform values are not correct. Please check your FHT calculation and rounding.</p>
            <p><strong>Hint:</strong> Remember the FHT butterfly stages:
            <ul>
                <li>Stage 1: \(a_0 = y_0+y_1\), \(a_1 = y_0-y_1\), \(a_2 = y_2+y_3\), \(a_3 = y_2-y_3\)</li>
                <li>Stage 2: \(z_0 = a_0+a_2\), \(z_1 = a_1+a_3\), \(z_2 = a_0-a_2\), \(z_3 = a_1-a_3\)</li>
            </ul>
            </p>
        `;
    }
    
    typesetMath();
}

function checkPart2() {
    // 1. Get user inputs
    const userCodeword = [
        parseInt(document.getElementById('c0').value),
        parseInt(document.getElementById('c1').value),
        parseInt(document.getElementById('c2').value),
        parseInt(document.getElementById('c3').value)
    ];
    
    // 2. Check for empty inputs
    if (userCodeword.some(isNaN)) {
        alert("Please fill in all 4 boxes for Part 2.");
        return;
    }
    
    // 3. Compare user's codeword to the *true* ML codeword
    let codewordCorrect = (
        userCodeword[0] === solution.mlCodeword[0] &&
        userCodeword[1] === solution.mlCodeword[1] &&
        userCodeword[2] === solution.mlCodeword[2] &&
        userCodeword[3] === solution.mlCodeword[3]
    );

    // 4. Generate feedback
    const feedbackEl = document.getElementById('observation');
    // We must get the innerHTML *before* overwriting it
    const part1Feedback = document.getElementById('observation').innerHTML; 
    
    const basisVectorName = `\\(\\mathbf{h}_{${solution.k_ml}}\\)`;
    const basisVectorSign = (solution.m_hat_0 === 0) ? '+' : '-';

    if (codewordCorrect) {
        feedbackEl.className = 'obs-correct';
        feedbackEl.innerHTML = `
            ${part1Feedback}
            <hr>
            <h4>Part 2 Correct!</h4>
            <p>Your estimate \(\\mathbf{\\hat{c}} = (${userCodeword.join(', ')})\) is the correct ML codeword.</p>
            <p><strong>Reasoning:</strong>
            <ul>
                <li>The transform was \(\\mathbf{z} = (${solution.roundedTransformZ.join(', ')})\).</li>
                <li>The largest absolute value was \(|z_{${solution.k_ml}}| = ${Math.abs(solution.roundedTransformZ[solution.k_ml])}\) (from index \(k=${solution.k_ml}\)).</li>
                <li>This selects the basis vector \(${basisVectorSign}${basisVectorName}\).</li>
                <li>This maps to the ML message \((\\hat{m}_0, \\hat{m}_1, \\hat{m}_2) = (${solution.mlMessage.join(', ')})\).</li>
                <li>Reconstructing gives \(\\mathbf{\\hat{c}} = (${solution.mlCodeword.join(', ')})\).</li>
            </ul>
            </p>
        `;
    } else {
        feedbackEl.className = 'obs-partial'; // Partial, since Part 1 was correct
        feedbackEl.innerHTML = `
            ${part1Feedback}
            <hr>
            <h4>Part 2 Incorrect.</h4>
            <p>Your estimate \(\\mathbf{\\hat{c}} = (${userCodeword.join(', ')})\) is not the correct ML codeword.</p>
            <p><strong>Hint:</strong> Re-check the ML decoding steps:
            <ul>
                <li>Find the index \(k \in \\{0, 1, 2, 3\\}\) with the largest \(|z_k|\).</li>
                <li>Use this \(k\) to find \((\hat{m}_1, \hat{m}_2)\). (e.g., \(k=1 \implies (1,0)\))</li>
                <li>Use the sign of \(z_k\) to find \(\hat{m}_0\). (\(z_k \le 0 \implies \hat{m}_0=1\))</li>
                <li>Reconstruct \(\\mathbf{\\hat{c}} = \\hat{m}_0 \\mathbf{v}_0 \oplus \\hat{m}_1 \\mathbf{v}_1 \oplus \\hat{m}_2 \\mathbf{v}_2\).</li>
            </ul>
            </p>
        `;
    }
    
    typesetMath();
}