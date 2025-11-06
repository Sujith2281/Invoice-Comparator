# 🧾 Invoice Comparator Pro

A powerful Django-based web application for intelligent invoice analysis and comparison using advanced OCR technology and fuzzy matching algorithms.

# purpose
it is a application that helps to compare client & vendor side invoices (cross verification)

## ✨ Features

### 📤 **Invoice Processing**
- **Multi-format Support**: Upload PDF files and images (PNG, JPG, etc.)
- **Advanced OCR**: Powered by PaddleOCR for accurate text extraction
- **Smart Table Detection**: Automatically identifies billing and summary sections
- **Data Structuring**: Converts unstructured invoice data into organized tables

### 🔍 **Intelligent Comparison**
- **Side-by-Side Analysis**: Compare two invoices with visual highlighting
- **Fuzzy Matching**: Smart product/service matching across different invoice formats
- **Column Mapping**: Flexible column comparison with customizable field selection
- **Difference Detection**: Automatic identification of mismatched data with color coding

### 📊 **Analysis Tools**
- **Field Comparison**: Compare specific fields like amounts, quantities, descriptions
- **Custom Column Selection**: Create custom comparison tables with selected columns
- **Unified Table Generation**: Merge data from both invoices into a single view
- **Match Status Tracking**: Visual indicators for matched, unmatched, and missing items

### 🎨 **User Experience**
- **Modern UI**: Clean, responsive design with gradient backgrounds
- **Scrollable Tables**: Horizontal and vertical scrolling for large datasets
- **Interactive Elements**: Clickable cells with hover effects
- **Mobile Responsive**: Optimized for desktop and mobile devices

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Django 5.2+
- CUDA-compatible GPU (recommended for OCR performance)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/invoice-comparator-pro.git
cd invoice-comparator-pro
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Navigate to Django project**
```bash
cd invoiceHandler
```

4. **Run migrations**
```bash
python manage.py migrate
```

5. **Start the development server**
```bash
python manage.py runserver
```

6. **Access the application**
Open your browser and go to `http://localhost:8000`

## 📋 Usage

### Basic Workflow

1. **Upload Invoices**: Select two PDF/image files using the upload interface
2. **Automatic Processing**: The system extracts and structures data using OCR
3. **View Raw Data**: Examine extracted billing and summary tables
4. **Compare Fields**: Use the comparison tabs to analyze specific fields
5. **Custom Analysis**: Create custom column combinations for detailed comparison

### Comparison Modes

#### 🔍 **Field Compare**
- Select predefined fields (Name, Date, Description, Quantity, Unit Price, Amount, Total)
- View side-by-side comparison with difference highlighting
- Automatic detection of mismatched values

#### 📋 **Column Select**
- Choose specific columns from each invoice
- Create multiple column pair combinations
- Generate comparison tables with match status indicators

### Advanced Features

#### **Smart Matching**
- Fuzzy string matching for product names and descriptions
- Numeric comparison with tolerance for amounts
- Automatic organization and date extraction

#### **Data Export**
- View structured data in clean table format
- Scrollable interface for large datasets
- Color-coded differences for easy identification

## 🛠️ Technical Architecture

### Backend Components

#### **OCR Engine** (`extractor.py`)
- **PaddleOCR Integration**: High-accuracy text recognition
- **Image Processing**: PDF to image conversion with OpenCV
- **Table Structure Detection**: Smart identification of billing tables
- **Data Cleaning**: Normalization and structuring of extracted text

#### **Comparison Engine** (`views.py`)
- **Fuzzy Matching**: Uses FuzzyWuzzy for intelligent text comparison
- **Column Mapping**: Automatic and manual column association
- **Data Transformation**: Pandas-based data manipulation
- **API Endpoints**: RESTful APIs for frontend communication

#### **Data Models**
- **Session Management**: Temporary storage of invoice data
- **Column Mapping**: Flexible field association system
- **Comparison Results**: Structured output for frontend rendering

### Frontend Components

#### **Modern UI** (`home.html`, `home.css`)
- **Responsive Design**: Grid-based layout with mobile optimization
- **Interactive Tables**: Scrollable containers with custom scrollbars
- **Visual Feedback**: Color-coded differences and match indicators
- **Tab Navigation**: Organized interface for different comparison modes

#### **Dynamic JavaScript** (`home.js`)
- **AJAX Communication**: Seamless backend integration
- **Table Rendering**: Dynamic table generation and updates
- **User Interactions**: Click handlers and form submissions
- **Data Visualization**: Real-time comparison result display

## 📁 Project Structure

```
project_invoice/
├── invoiceHandler/                 # Django project root
│   ├── invoice/                   # Main application
│   │   ├── static/invoice/        # Static files (CSS, JS)
│   │   ├── templates/invoice/     # HTML templates
│   │   ├── extractor.py          # OCR and data extraction
│   │   ├── views.py              # API endpoints and logic
│   │   ├── urls.py               # URL routing
│   │   └── models.py             # Data models
│   ├── invoiceHandler/           # Project settings
│   │   ├── settings.py           # Django configuration
│   │   ├── urls.py               # Main URL configuration
│   │   └── wsgi.py               # WSGI configuration
│   └── manage.py                 # Django management script
├── requirements.txt              # Python dependencies
├── invoices.yml                  # Conda environment file
└── README.md                     # Project documentation
```

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main application interface |
| `/upload/` | POST | Upload and process invoices |
| `/column-comparator/` | POST | Compare specific columns |
| `/get-dropdown-columns/` | GET | Get available columns |
| `/compare-selected-columns/` | POST | Compare selected column pairs |
| `/final-result-mapper/` | POST | Generate final comparison results |

## 🎯 Key Features Explained

### **Intelligent OCR Processing**
- Converts PDF/images to structured data
- Handles various invoice formats and layouts
- Extracts billing tables, summary sections, and metadata
- Cleans and normalizes extracted text

### **Advanced Comparison Algorithms**
- **Fuzzy String Matching**: Handles variations in product names
- **Numeric Tolerance**: Accounts for rounding differences in amounts
- **Column Mapping**: Automatically maps similar columns across invoices
- **Missing Data Handling**: Gracefully handles incomplete data

### **Responsive Table Interface**
- **Horizontal Scrolling**: Handles wide tables with many columns
- **Vertical Scrolling**: Manages long lists of items
- **Custom Scrollbars**: Styled scrollbars matching the application theme
- **Mobile Optimization**: Responsive design for all screen sizes

## 🔍 Troubleshooting

### Common Issues

1. **OCR Performance**: Ensure CUDA is properly installed for GPU acceleration
2. **Large Files**: For very large invoices, processing may take longer
3. **Complex Layouts**: Some invoice formats may require manual column mapping
4. **Memory Usage**: Large datasets may require increased system memory

### Performance Tips

- Use high-quality, clear invoice images for better OCR results
- Ensure invoices are properly oriented (not rotated)
- For best results, use invoices with clear table structures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PaddleOCR** for excellent OCR capabilities
- **Django** for the robust web framework
- **FuzzyWuzzy** for intelligent string matching
- **OpenCV** for image processing capabilities

## 📞 Support

For support, please open an issue on GitHub or contact the development team.

---

**Built with ❤️ using Django, PaddleOCR, and modern web technologies**
