// home.js

let billing1Data = [];
let billing2Data = [];
let summary1Data = [];
let summary2Data = [];
let selectedColumnPairs = [];

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("invoice-form");
  const status = document.getElementById("status");
  const billing1Table = document.getElementById("billing1");
  const billing2Table = document.getElementById("billing2");
  const summary1table = document.getElementById("summary1");
  const summary2table = document.getElementById("summary2");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    status.textContent = "Uploading and processing...";

    const formData = new FormData(form);
    const csrfToken = document.querySelector('meta[name="csrf-token"]').content;

    try {
      const response = await fetch("/upload/", {
        method: "POST",
        headers: {
          "X-CSRFToken": csrfToken
        },
        body: formData
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || "Upload failed");
      }

      const data = await response.json();
      billing1Data = data.billing1 || [];
      billing2Data = data.billing2 || [];
      summary1Data = data.summary1 || [];
      summary2Data = data.summary2 || [];

      renderTable(billing1Data, billing1Table, "billing1");
      renderTable(billing2Data, billing2Table, "billing2");
      renderTable(summary1Data, summary1table, "summary1");
      renderTable(summary2Data, summary2table, "summary2");
      
      status.textContent = "";
      document.getElementById("analysis-tabs").style.display = "block";
      document.getElementById("raw-data-section").style.display = "block";
      
      switchTab('column-select');
      populateColumnDropdown();
      populateDropdownColumns();
      
    } catch (err) {
      status.textContent = "Error: " + err.message;
    }
  });

  // Tab switching
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tabName = btn.getAttribute('data-tab');
      switchTab(tabName);
    });
  });

  // Data toggle for billing/summary
  document.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.getAttribute('data-target');
      
      document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.data-content').forEach(c => c.classList.remove('active'));
      
      btn.classList.add('active');
      document.getElementById(target).classList.add('active');
    });
  });

  // Add comparison button
  document.getElementById("add-comparison-btn").addEventListener("click", function() {
    const column1 = document.getElementById("invoice1-dropdown").value;
    const column2 = document.getElementById("invoice2-dropdown").value;
    
    if (!column1 || !column2) {
      alert("Please select columns from both invoices");
      return;
    }
    
    selectedColumnPairs.push({
      column1: column1,
      column2: column2,
      display_name: `${column1} + ${column2}`
    });
    
    document.getElementById("invoice1-dropdown").value = "";
    document.getElementById("invoice2-dropdown").value = "";
    
    updateSelectedCombinationsList();
  });

  // Create table button
  document.getElementById("create-table-btn").addEventListener("click", async function() {
    if (selectedColumnPairs.length === 0) {
      alert("Please add at least one column combination");
      return;
    }
    
    await fetchMultipleColumnComparison();
  });

  // Remove button event delegation
  document.addEventListener('click', function(e) {
    if (e.target.classList.contains('remove-btn')) {
      const index = parseInt(e.target.getAttribute('data-index'));
      removeColumnPair(index);
    }
  });
});

function renderTable(data, tableElement, tableId) {
  tableElement.innerHTML = "";
  if (data.length === 0) {
    tableElement.innerHTML = "<tr><td>No data available</td></tr>";
    return;
  }

  const headers = Object.keys(data[0]);
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  headers.forEach(header => {
    const th = document.createElement("th");
    th.textContent = header;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  tableElement.appendChild(thead);

  const tbody = document.createElement("tbody");
  data.forEach((row, rowIndex) => {
    const tr = document.createElement("tr");
    headers.forEach((key, colIndex) => {
      const td = document.createElement("td");
      td.textContent = row[key];
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  tableElement.appendChild(tbody);
}

function populateColumnDropdown() {
  const fixedColumns = [
    { value: "name", text: "Name" },
    { value: "date", text: "Invoice Date" },
    { value: "description", text: "Description/Product" },
    { value: "qty", text: "Quantity" },
    { value: "unit", text: "Unit Price" },
    { value: "amount", text: "Amount" }, 
    { value: "total", text: "Total Amount" }
  ];

  const dropdown = document.getElementById("column-dropdown");
  if (!dropdown) return;
  
  dropdown.innerHTML = "";
  
  fixedColumns.forEach(col => {
    const option = document.createElement("option");
    option.value = col.value;
    option.textContent = col.text;
    dropdown.appendChild(option);
  });
  
  dropdown.addEventListener("change", async function() {
    const selectedColumn = this.value;
    await fetchAndRenderColumnComparison(selectedColumn);
  });
}

async function fetchAndRenderColumnComparison(selectedCol) {
  const csrfToken = document.querySelector('meta[name="csrf-token"]').content;
  const table = document.getElementById("column-comparison-table");
  table.innerHTML = "<tr><td>Loading...</td></tr>";

  try {
    const response = await fetch("/column-comparator/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": csrfToken
      },
      body: JSON.stringify({ column: selectedCol })
    });

    const data = await response.json();

    if (data.error) {
      table.innerHTML = `<tr><td>${data.error}</td></tr>`;
      return;
    }

    table.innerHTML = "";
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    headerRow.appendChild(document.createElement("th")).textContent = data.selected_column;
    headerRow.appendChild(document.createElement("th")).textContent = "Invoice 1";
    headerRow.appendChild(document.createElement("th")).textContent = "Invoice 2";
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");

    if (selectedCol === "name") {
      const org1 = data.organization1 || [];
      const org2 = data.organization2 || [];
      const maxLen = Math.max(org1.length, org2.length);
      
      for (let i = 0; i < maxLen; i++) {
        const tr = document.createElement("tr");
        const val1 = org1[i] || "No Data";
        const val2 = org2[i] || "No Data";
        const isDiff = val1 !== val2;
        
        tr.appendChild(document.createElement("td")).textContent = `Organization ${i + 1}`;
        const td1 = document.createElement("td");
        td1.textContent = val1;
        if (isDiff) td1.classList.add("cell-diff");
        tr.appendChild(td1);
        
        const td2 = document.createElement("td");
        td2.textContent = val2;
        if (isDiff) td2.classList.add("cell-diff");
        tr.appendChild(td2);
        
        tbody.appendChild(tr);
      }
    } else if (selectedCol === "date") {
      const date1 = data.date1 || [];
      const date2 = data.date2 || [];
      const maxLen = Math.max(date1.length, date2.length);
      
      for (let i = 0; i < maxLen; i++) {
        const tr = document.createElement("tr");
        const val1 = date1[i] || "No Data";
        const val2 = date2[i] || "No Data";
        const isDiff = val1 !== val2;
        
        tr.appendChild(document.createElement("td")).textContent = `Date ${i + 1}`;
        const td1 = document.createElement("td");
        td1.textContent = val1;
        if (isDiff) td1.classList.add("cell-diff");
        tr.appendChild(td1);
        
        const td2 = document.createElement("td");
        td2.textContent = val2;
        if (isDiff) td2.classList.add("cell-diff");
        tr.appendChild(td2);
        
        tbody.appendChild(tr);
      }
    } else {
      data.comparison_result.forEach((row, index) => {
        const tr = document.createElement("tr");
        
        tr.appendChild(document.createElement("td")).textContent = `Row ${index + 1}`;
        
        const td1 = document.createElement("td");
        td1.textContent = row.invoice1 || "No Data";
        if (row.diff) td1.classList.add("cell-diff");
        tr.appendChild(td1);
        
        const td2 = document.createElement("td");
        td2.textContent = row.invoice2 || "No Data";
        if (row.diff) td2.classList.add("cell-diff");
        tr.appendChild(td2);
        
        tbody.appendChild(tr);
      });
    }

    table.appendChild(tbody);

  } catch (err) {
    table.innerHTML = `<tr><td>Error loading comparison</td></tr>`;
  }
}

async function populateDropdownColumns() {
  try {
    const response = await fetch("/get-dropdown-columns/");
    const data = await response.json();
    
    if (data.error) {
      console.error("Error fetching columns:", data.error);
      return;
    }
    
    const invoice1Dropdown = document.getElementById("invoice1-dropdown");
    const invoice2Dropdown = document.getElementById("invoice2-dropdown");
    
    invoice1Dropdown.innerHTML = '<option value="">Select Column</option>';
    invoice2Dropdown.innerHTML = '<option value="">Select Column</option>';
    
    data.invoice1_columns.forEach(column => {
      const option = document.createElement("option");
      option.value = column;
      option.textContent = column;
      invoice1Dropdown.appendChild(option);
    });
    
    data.invoice2_columns.forEach(column => {
      const option = document.createElement("option");
      option.value = column;
      option.textContent = column;
      invoice2Dropdown.appendChild(option);
    });
    
  } catch (err) {
    console.error("Error populating dropdown columns:", err);
  }
}

function updateSelectedCombinationsList() {
  const listDiv = document.getElementById("selected-combinations-list");
  const list = document.getElementById("combinations-list");
  
  list.innerHTML = "";
  selectedColumnPairs.forEach((pair, index) => {
    const li = document.createElement("li");
    li.innerHTML = `${pair.display_name} <button class="remove-btn" data-index="${index}">Remove</button>`;
    list.appendChild(li);
  });
  
  listDiv.style.display = selectedColumnPairs.length > 0 ? "block" : "none";
}

function removeColumnPair(index) {
  selectedColumnPairs.splice(index, 1);
  updateSelectedCombinationsList();
}

async function fetchMultipleColumnComparison() {
  const csrfToken = document.querySelector('meta[name="csrf-token"]').content;
  const tablesContainer = document.getElementById("tables-container");
  const resultDiv = document.getElementById("dropdown-comparison-result");
  
  tablesContainer.innerHTML = "<div>Loading...</div>";
  resultDiv.style.display = "block";
  
  try {
    const response = await fetch("/compare-selected-columns/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": csrfToken
      },
      body: JSON.stringify({ column_pairs: selectedColumnPairs })
    });
    
    const data = await response.json();
    
    if (data.error) {
      tablesContainer.innerHTML = `<div>${data.error}</div>`;
      return;
    }
    
    tablesContainer.innerHTML = "";
    
    data.column_pairs.forEach((pair, index) => {
      const tableWrapper = document.createElement("div");
      tableWrapper.className = "table-wrapper-horizontal";
      
      const tableTitle = document.createElement("h4");
      tableTitle.textContent = `${pair.column1} vs ${pair.column2}`;
      tableWrapper.appendChild(tableTitle);
      
      const table = document.createElement("table");
      table.className = "comparison-table";
      
      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      
      const th1 = document.createElement("th");
      th1.textContent = `${pair.column1} (Invoice 1)`;
      headerRow.appendChild(th1);
      
      const th2 = document.createElement("th");
      th2.textContent = `${pair.column2} (Invoice 2)`;
      headerRow.appendChild(th2);
      
      const th3 = document.createElement("th");
      th3.textContent = "Status";
      headerRow.appendChild(th3);
      
      thead.appendChild(headerRow);
      table.appendChild(thead);
      
      const tbody = document.createElement("tbody");
      data.comparison_data.forEach(rowData => {
        const tr = document.createElement("tr");
        
        const col1Key = `${pair.column1}_invoice1`;
        const col2Key = `${pair.column2}_invoice2`;
        const matchKey = `${pair.column1}_${pair.column2}_match`;
        
        const td1 = document.createElement("td");
        td1.textContent = rowData[col1Key] || "";
        tr.appendChild(td1);
        
        const td2 = document.createElement("td");
        td2.textContent = rowData[col2Key] || "";
        tr.appendChild(td2);
        
        const td3 = document.createElement("td");
        td3.textContent = rowData[matchKey] || "";
        if (rowData[matchKey] === 'no') {
          td3.style.color = 'red';
          td3.style.fontWeight = 'bold';
        } else if (rowData[matchKey] === 'yes') {
          td3.style.color = 'green';
          td3.style.fontWeight = 'bold';
        }
        tr.appendChild(td3);
        
        tbody.appendChild(tr);
      });
      
      table.appendChild(tbody);
      tableWrapper.appendChild(table);
      tablesContainer.appendChild(tableWrapper);
    });
    
  } catch (err) {
    tablesContainer.innerHTML = `<div>Error loading comparison</div>`;
    console.error("Error fetching multiple column comparison:", err);
  }
}

function switchTab(tabName) {
  document.querySelectorAll('.tab-content').forEach(tab => {
    tab.classList.remove('active');
  });
  
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  
  document.getElementById(`${tabName}-tab`).classList.add('active');
  document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
}