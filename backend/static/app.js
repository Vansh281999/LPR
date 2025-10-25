// Frontend JS to wire API calls
const imageInput = document.getElementById('imageInput');
const detectBtn = document.getElementById('detectBtn');
const dropzone = document.getElementById('dropzone');
const previewImg = document.getElementById('previewImg');
const uploadPreview = document.getElementById('uploadPreview');
const weatherSelect = document.getElementById('weatherSelect');

const resultsEmpty = document.getElementById('resultsEmpty');
const resultsBlock = document.getElementById('results');
const plateTextEl = document.getElementById('plateText');
const confidenceTextEl = document.getElementById('confidenceText');
const accuracyTextEl = document.getElementById('accuracyText');
const authorizedTextEl = document.getElementById('authorizedText');
const ownerBlock = document.getElementById('ownerBlock');
const ownerNameEl = document.getElementById('ownerName');
const ownerAptEl = document.getElementById('ownerApt');
const ownerPhoneEl = document.getElementById('ownerPhone');

const accuracyValue = document.getElementById('accuracyValue');
const detectionsValue = document.getElementById('detectionsValue');
const registeredValue = document.getElementById('registeredValue');
const notificationsValue = document.getElementById('notificationsValue');

// Registry form
const registerForm = document.getElementById('registerForm');
const registerMsg = document.getElementById('registerMsg');

// Enable detect button when a file is chosen
imageInput.addEventListener('change', () => {
  const file = imageInput.files[0];
  if (file) {
    detectBtn.disabled = false;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    uploadPreview.hidden = false;
  }
});

// Clicking dropzone opens file selection
if (dropzone) {
  dropzone.addEventListener('click', () => imageInput.click());
}

// Detect license plate handler
if (detectBtn) {
  detectBtn.addEventListener('click', async () => {
    const file = imageInput.files[0];
    if (!file) return;

    detectBtn.disabled = true;
    detectBtn.textContent = 'Processing...';

    const formData = new FormData();
    formData.append('image', file);
    formData.append('weather_conditions', weatherSelect.value);

    try {
      const res = await fetch('/api/process-image', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();

      detectBtn.disabled = false;
      detectBtn.textContent = 'Detect License Plate';

      if (!res.ok) {
        resultsEmpty.textContent = data.error || 'Detection failed';
        resultsEmpty.hidden = false;
        resultsBlock.hidden = true;
        return;
      }

      // Populate UI with results
      plateTextEl.textContent = data.plate_text ?? '-';
      confidenceTextEl.textContent = data.confidence ? `${(data.confidence * 100).toFixed(1)}%` : '-';
      accuracyTextEl.textContent = data.accuracy !== null && data.accuracy !== undefined ? `${(data.accuracy * 100).toFixed(1)}%` : 'N/A';
      authorizedTextEl.textContent = data.is_authorized ? 'Yes' : 'No';

      if (data.owner_info && data.owner_info.owner_name) {
        ownerBlock.hidden = false;
        ownerNameEl.textContent = data.owner_info.owner_name;
        ownerAptEl.textContent = data.owner_info.apartment_number || '-';
        ownerPhoneEl.textContent = data.owner_info.contact_phone || '-';
      } else {
        ownerBlock.hidden = true;
      }

      resultsEmpty.hidden = true;
      resultsBlock.hidden = false;

      // Refresh stats after each detection
      loadStats();
    } catch (err) {
      console.error(err);
      resultsEmpty.textContent = 'Unexpected error while processing.';
      resultsEmpty.hidden = false;
      resultsBlock.hidden = true;
      detectBtn.disabled = false;
      detectBtn.textContent = 'Detect License Plate';
    }
  });
}

// Load performance stats and totals
async function loadStats() {
  try {
    const res = await fetch('/api/performance-stats');
    const stats = await res.json();
    const avgAcc = stats.average_accuracy ?? 0;
    accuracyValue.textContent = `${(avgAcc * 100).toFixed(1)}%`;
    detectionsValue.textContent = stats.total_detections ?? 0;
    registeredValue.textContent = stats.authorized_vehicles ?? 0; // proxy for registered
    notificationsValue.textContent = stats.authorized_vehicles ?? 0; // placeholder
  } catch (e) {
    console.error('Failed to load stats', e);
  }
}

// Registry form submit
if (registerForm) {
  registerForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    registerMsg.textContent = '';

    const payload = Object.fromEntries(new FormData(registerForm));
    try {
      const res = await fetch('/api/register-vehicle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        registerMsg.textContent = 'Vehicle registered successfully';
        registerMsg.style.color = '#93c5fd';
        registerForm.reset();
        loadStats();
      } else {
        registerMsg.textContent = data.error || 'Failed to register';
        registerMsg.style.color = '#fca5a5';
      }
    } catch (err) {
      registerMsg.textContent = 'Unexpected error';
      registerMsg.style.color = '#fca5a5';
    }
  });
}

// Initial load
loadStats();