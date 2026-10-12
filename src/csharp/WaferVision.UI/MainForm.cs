using System;
using System.Drawing;
using System.Windows.Forms;
using OpenCvSharp;
using ScottPlot;

namespace WaferVision.UI
{
    public partial class MainForm : Form
    {
        private PictureBox waferPictureBox;
        private PictureBox defectPictureBox;
        private Button loadButton;
        private Button processButton;
        private Button alignButton;
        private Button detectButton;
        private Button metrologyButton;
        private TextBox logTextBox;
        private Label statusLabel;
        private ProgressBar progressBar;
        private TabControl mainTabControl;
        private TabPage imageTabPage;
        private TabPage metricsTabPage;
        private ScottPlot.FormsPlot metricsPlot;
        
        private Mat currentImage;
        private Mat processedImage;
        private Mat defectMap;
        
        public MainForm()
        {
            InitializeComponent();
            InitializeCustomComponents();
        }
        
        private void InitializeComponent()
        {
            this.SuspendLayout();
            
            // Form
            this.Text = "WaferVision - Semiconductor Inspection System";
            this.Size = new Size(1400, 900);
            this.StartPosition = FormStartPosition.CenterScreen;
            this.MinimumSize = new Size(1200, 800);
            
            this.ResumeLayout(false);
        }
        
        private void InitializeCustomComponents()
        {
            // Main Tab Control
            mainTabControl = new TabControl();
            mainTabControl.Dock = DockStyle.Fill;
            mainTabControl.Location = new Point(0, 0);
            mainTabControl.Size = new Size(1400, 700);
            
            // Image Tab Page
            imageTabPage = new TabPage("Image Processing");
            
            // Wafer Image Display
            waferPictureBox = new PictureBox();
            waferPictureBox.Location = new Point(20, 20);
            waferPictureBox.Size = new Size(600, 600);
            waferPictureBox.BorderStyle = BorderStyle.FixedSingle;
            waferPictureBox.SizeMode = PictureBoxSizeMode.Zoom;
            waferPictureBox.BackColor = Color.Black;
            
            // Defect Map Display
            defectPictureBox = new PictureBox();
            defectPictureBox.Location = new Point(640, 20);
            defectPictureBox.Size = new Size(600, 600);
            defectPictureBox.BorderStyle = BorderStyle.FixedSingle;
            defectPictureBox.SizeMode = PictureBoxSizeMode.Zoom;
            defectPictureBox.BackColor = Color.Black;
            
            imageTabPage.Controls.Add(waferPictureBox);
            imageTabPage.Controls.Add(defectPictureBox);
            
            // Metrics Tab Page
            metricsTabPage = new TabPage("Inspection Metrics");
            
            metricsPlot = new ScottPlot.FormsPlot();
            metricsPlot.Location = new Point(20, 20);
            metricsPlot.Size = new Size(1220, 600);
            
            metricsTabPage.Controls.Add(metricsPlot);
            
            mainTabControl.TabPages.Add(imageTabPage);
            mainTabControl.TabPages.Add(metricsTabPage);
            
            // Control Panel
            Panel controlPanel = new Panel();
            controlPanel.Dock = DockStyle.Bottom;
            controlPanel.Height = 150;
            controlPanel.BackColor = Color.LightGray;
            
            // Buttons
            loadButton = new Button();
            loadButton.Text = "Load Wafer Image";
            loadButton.Location = new Point(20, 20);
            loadButton.Size = new Size(150, 40);
            loadButton.Click += LoadButton_Click;
            
            processButton = new Button();
            processButton.Text = "Preprocess";
            processButton.Location = new Point(180, 20);
            processButton.Size = new Size(150, 40);
            processButton.Click += ProcessButton_Click;
            
            alignButton = new Button();
            alignButton.Text = "Align Wafer";
            alignButton.Location = new Point(340, 20);
            alignButton.Size = new Size(150, 40);
            alignButton.Click += AlignButton_Click;
            
            detectButton = new Button();
            detectButton.Text = "Detect Defects";
            detectButton.Location = new Point(500, 20);
            detectButton.Size = new Size(150, 40);
            detectButton.Click += DetectButton_Click;
            
            metrologyButton = new Button();
            metrologyButton.Text = "Run Metrology";
            metrologyButton.Location = new Point(660, 20);
            metrologyButton.Size = new Size(150, 40);
            metrologyButton.Click += MetrologyButton_Click;
            
            // Progress Bar
            progressBar = new ProgressBar();
            progressBar.Location = new Point(20, 80);
            progressBar.Size = new Size(400, 25);
            
            // Status Label
            statusLabel = new Label();
            statusLabel.Text = "Ready";
            statusLabel.Location = new Point(20, 115);
            statusLabel.Size = new Size(400, 20);
            
            // Log TextBox
            logTextBox = new TextBox();
            logTextBox.Location = new Point(440, 70);
            logTextBox.Size = new Size(900, 70);
            logTextBox.Multiline = true;
            logTextBox.ScrollBars = ScrollBars.Vertical;
            logTextBox.ReadOnly = true;
            logTextBox.BackColor = Color.White;
            
            controlPanel.Controls.Add(loadButton);
            controlPanel.Controls.Add(processButton);
            controlPanel.Controls.Add(alignButton);
            controlPanel.Controls.Add(detectButton);
            controlPanel.Controls.Add(metrologyButton);
            controlPanel.Controls.Add(progressBar);
            controlPanel.Controls.Add(statusLabel);
            controlPanel.Controls.Add(logTextBox);
            
            this.Controls.Add(mainTabControl);
            this.Controls.Add(controlPanel);
        }
        
        private void LoadButton_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Image Files|*.png;*.jpg;*.jpeg;*.bmp;*.tiff|All Files|*.*";
            openFileDialog.Title = "Select Wafer Image";
            
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    currentImage = Cv2.ImRead(openFileDialog.FileName, ImreadModes.Grayscale);
                    DisplayImage(currentImage, waferPictureBox);
                    LogMessage($"Loaded image: {openFileDialog.FileName}");
                    LogMessage($"Image size: {currentImage.Width}x{currentImage.Height}");
                    statusLabel.Text = "Image loaded";
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Error loading image: {ex.Message}", "Error", 
                                  MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }
        
        private void ProcessButton_Click(object sender, EventArgs e)
        {
            if (currentImage == null)
            {
                MessageBox.Show("Please load an image first.", "Warning", 
                              MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            
            progressBar.Value = 0;
            statusLabel.Text = "Preprocessing...";
            
            try
            {
                // Denoise
                Mat denoised = new Mat();
                Cv2.FastNlMeansDenoising(currentImage, denoised, 10, 7, 21);
                progressBar.Value = 33;
                
                // Normalize
                Mat normalized = new Mat();
                Cv2.Normalize(denoised, normalized, 0, 255, NormTypes.MinMax, MatType.CV_8U);
                progressBar.Value = 66;
                
                // Enhance contrast
                Mat enhanced = new Mat();
                Cv2.EqualizeHist(normalized, enhanced);
                progressBar.Value = 100;
                
                processedImage = enhanced;
                DisplayImage(processedImage, waferPictureBox);
                
                LogMessage("Preprocessing completed");
                statusLabel.Text = "Preprocessing complete";
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error during preprocessing: {ex.Message}", "Error", 
                              MessageBoxButtons.OK, MessageBoxIcon.Error);
                statusLabel.Text = "Preprocessing failed";
            }
        }
        
        private void AlignButton_Click(object sender, EventArgs e)
        {
            if (currentImage == null)
            {
                MessageBox.Show("Please load an image first.", "Warning", 
                              MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            
            statusLabel.Text = "Aligning wafer...";
            
            try
            {
                Mat imageToAlign = processedImage ?? currentImage;
                
                // Simple center alignment
                Mat gray = new Mat();
                if (imageToAlign.Channels() == 3)
                {
                    Cv2.CvtColor(imageToAlign, gray, ColorConversionCodes.BGR2GRAY);
                }
                else
                {
                    gray = imageToAlign.Clone();
                }
                
                // Find center using moments
                Mat binary = new Mat();
                Cv2.Threshold(gray, binary, 0, 255, ThresholdTypes.Otsu | ThresholdTypes.Binary);
                
                Moments m = Cv2.Moments(binary);
                Point2f center = new Point2f((float)(m.M10 / m.M00), (float)(m.M01 / m.M00));
                
                // Calculate translation to center
                Point2f imageCenter = new Point2f(gray.Width / 2f, gray.Height / 2f);
                Point2f translation = new Point2f(imageCenter.X - center.X, imageCenter.Y - center.Y);
                
                Mat translationMatrix = Mat.FromArray(new double[,] {
                    { 1, 0, translation.X },
                    { 0, 1, translation.Y },
                    { 0, 0, 1 }
                });
                
                Mat aligned = new Mat();
                Cv2.WarpPerspective(imageToAlign, aligned, translationMatrix, imageToAlign.Size());
                
                processedImage = aligned;
                DisplayImage(processedImage, waferPictureBox);
                
                LogMessage($"Wafer aligned. Center offset: ({translation.X:F1}, {translation.Y:F1})");
                statusLabel.Text = "Alignment complete";
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error during alignment: {ex.Message}", "Error", 
                              MessageBoxButtons.OK, MessageBoxIcon.Error);
                statusLabel.Text = "Alignment failed";
            }
        }
        
        private void DetectButton_Click(object sender, EventArgs e)
        {
            if (currentImage == null)
            {
                MessageBox.Show("Please load an image first.", "Warning", 
                              MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            
            progressBar.Value = 0;
            statusLabel.Text = "Detecting defects...";
            
            try
            {
                Mat imageToProcess = processedImage ?? currentImage;
                
                // Edge detection for defect detection
                Mat edges = new Mat();
                Cv2.Canny(imageToProcess, edges, 50, 150);
                progressBar.Value = 25;
                
                // Find contours
                OpenCvSharp.Point[][] contours;
                HierarchyIndex[] hierarchy;
                Cv2.FindContours(edges, out contours, out hierarchy, 
                                RetrievalModes.External, ContourApproximationModes.ApproxSimple);
                progressBar.Value = 50;
                
                // Create defect map
                defectMap = new Mat();
                Cv2.CvtColor(imageToProcess, defectMap, ColorConversionCodes.GRAY2BGR);
                
                int defectCount = 0;
                foreach (var contour in contours)
                {
                    double area = Cv2.ContourArea(contour);
                    if (area > 10 && area < 1000)
                    {
                        Rect bbox = Cv2.BoundingRect(contour);
                        Cv2.Rectangle(defectMap, bbox, Scalar.Red, 2);
                        defectCount++;
                    }
                }
                progressBar.Value = 100;
                
                DisplayImage(defectMap, defectPictureBox);
                
                LogMessage($"Defect detection completed. Found {defectCount} defects.");
                statusLabel.Text = $"Found {defectCount} defects";
                
                // Update metrics plot
                UpdateMetricsPlot(defectCount);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error during defect detection: {ex.Message}", "Error", 
                              MessageBoxButtons.OK, MessageBoxIcon.Error);
                statusLabel.Text = "Defect detection failed";
            }
        }
        
        private void MetrologyButton_Click(object sender, EventArgs e)
        {
            if (currentImage == null)
            {
                MessageBox.Show("Please load an image first.", "Warning", 
                              MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            
            statusLabel.Text = "Running metrology...";
            
            try
            {
                Mat imageToProcess = processedImage ?? currentImage;
                
                // Calculate focus metric (sharpness)
                Mat laplacian = new Mat();
                Cv2.Laplacian(imageToProcess, laplacian, MatType.CV_64F);
                Scalar mean, stddev;
                Cv2.MeanStdDev(laplacian, out mean, out stddev);
                double sharpness = stddev.Val0 * stddev.Val0;
                
                // Calculate contrast
                Scalar imgMean, imgStddev;
                Cv2.MeanStdDev(imageToProcess, out imgMean, out imgStddev);
                double contrast = imgStddev.Val0;
                
                LogMessage($"Metrology results:");
                LogMessage($"  Sharpness metric: {sharpness:F2}");
                LogMessage($"  Contrast metric: {contrast:F2}");
                
                statusLabel.Text = "Metrology complete";
                
                // Update metrics plot with metrology data
                UpdateMetrologyPlot(sharpness, contrast);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error during metrology: {ex.Message}", "Error", 
                              MessageBoxButtons.OK, MessageBoxIcon.Error);
                statusLabel.Text = "Metrology failed";
            }
        }
        
        private void DisplayImage(Mat image, PictureBox pictureBox)
        {
            if (image.Empty()) return;
            
            Mat displayImage = new Mat();
            if (image.Channels() == 1)
            {
                Cv2.CvtColor(image, displayImage, ColorConversionCodes.GRAY2BGR);
            }
            else
            {
                displayImage = image.Clone();
            }
            
            Bitmap bitmap = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(displayImage);
            pictureBox.Image = bitmap;
        }
        
        private void LogMessage(string message)
        {
            logTextBox.AppendText($"[{DateTime.Now:HH:mm:ss}] {message}\r\n");
            logTextBox.ScrollToCaret();
        }
        
        private void UpdateMetricsPlot(int defectCount)
        {
            metricsPlot.Plot.Clear();
            
            double[] defectValues = { defectCount };
            string[] labels = { "Defects" };
            
            var bar = metricsPlot.Plot.AddBar(defectValues);
            bar.Label = "Defect Count";
            
            metricsPlot.Plot.XTicks(labels);
            metricsPlot.Plot.SetAxisLimits(yMin: 0);
            metricsPlot.Plot.Title("Defect Detection Results");
            metricsPlot.Plot.YLabel("Count");
            
            metricsPlot.Refresh();
        }
        
        private void UpdateMetrologyPlot(double sharpness, double contrast)
        {
            metricsPlot.Plot.Clear();
            
            double[] values = { sharpness, contrast };
            string[] labels = { "Sharpness", "Contrast" };
            
            var bar = metricsPlot.Plot.AddBar(values);
            bar.Label = "Metrology Metrics";
            
            metricsPlot.Plot.XTicks(labels);
            metricsPlot.Plot.Title("Metrology Results");
            metricsPlot.Plot.YLabel("Value");
            
            metricsPlot.Refresh();
        }
    }
}
