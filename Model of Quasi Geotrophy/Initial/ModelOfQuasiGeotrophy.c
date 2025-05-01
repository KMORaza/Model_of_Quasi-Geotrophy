#include <gtk/gtk.h>
#include <math.h>
#include <fftw3.h>
#include <string.h>

#ifdef MIN
#undef MIN
#endif
#define MIN(a,b) ((a) < (b) ? (a) : (b))

#define GRID_SIZE 64
#define DEFAULT_DT 0.01
#define DEFAULT_BETA 1.6e-11
#define DEFAULT_F0 1e-4
#define DEFAULT_G 9.81
#define DEFAULT_H 5000.0

// Model configuration structure
typedef struct {
    int layers; // Number of layers (2 or multi-layer)
    int grid_size; // Grid size
    double dt; // Time step
    double beta; // Planetary beta
    double f0; // Coriolis parameter
    double g; // Gravity
    double H; // Layer thickness
    char geometry[16]; // "beta-plane" or "spherical"
    char solver[16]; // "runge-kutta", "leapfrog", "semi-implicit"
    char boundary[16]; // "periodic", "rigid", "open"
    double wind_forcing; // Wind forcing amplitude
} QGConfig;

// Simulation data
typedef struct {
    int layers;
    int grid_size;
    double *psi; // Streamfunction
    double *q; // Potential vorticity
    double *u, *v; // Velocity components
    double *topography; // Bottom topography
    fftw_complex *psi_k, *q_k; // Fourier transforms
    fftw_plan fft_plan, ifft_plan;
} QGModel;

// GUI structure
typedef struct {
    GtkWidget *window;
    GtkWidget *config_box;
    GtkWidget *layers_spin;
    GtkWidget *geometry_dropdown;
    GtkWidget *solver_dropdown;
    GtkWidget *boundary_dropdown;
    GtkWidget *beta_entry;
    GtkWidget *dt_entry;
    GtkWidget *canvas;
    QGModel *model;
    QGConfig config;
    gboolean running;
} AppData;

// Initialize model
QGModel *qg_model_init(int layers, int grid_size) {
    QGModel *model = g_malloc(sizeof(QGModel));
    model->layers = layers;
    model->grid_size = grid_size;
    
    // Allocate arrays
    model->psi = g_malloc(grid_size * grid_size * layers * sizeof(double));
    model->q = g_malloc(grid_size * grid_size * layers * sizeof(double));
    model->u = g_malloc(grid_size * grid_size * layers * sizeof(double));
    model->v = g_malloc(grid_size * grid_size * layers * sizeof(double));
    model->topography = g_malloc(grid_size * grid_size * sizeof(double));
    model->psi_k = fftw_malloc(grid_size * grid_size * layers * sizeof(fftw_complex));
    model->q_k = fftw_malloc(grid_size * grid_size * layers * sizeof(fftw_complex));
    
    // Initialize FFTW plans (per layer)
    model->fft_plan = fftw_plan_dft_r2c_2d(grid_size, grid_size, model->psi, model->psi_k, FFTW_MEASURE);
    model->ifft_plan = fftw_plan_dft_c2r_2d(grid_size, grid_size, model->psi_k, model->psi, FFTW_MEASURE);
    
    // Initialize fields
    memset(model->psi, 0, grid_size * grid_size * layers * sizeof(double));
    memset(model->q, 0, grid_size * grid_size * layers * sizeof(double));
    memset(model->topography, 0, grid_size * grid_size * sizeof(double));
    
    // Initialize with a simple vortex for testing
    for (int l = 0; l < layers; l++) {
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                int idx = l * grid_size * grid_size + i * grid_size + j;
                double x = (i - grid_size / 2.0) / (grid_size / 4.0);
                double y = (j - grid_size / 2.0) / (grid_size / 4.0);
                model->psi[idx] = exp(-(x * x + y * y));
                model->q[idx] = model->psi[idx]; // Simplified initial PV
            }
        }
    }
    
    return model;
}

// Free model
void qg_model_free(QGModel *model) {
    fftw_destroy_plan(model->fft_plan);
    fftw_destroy_plan(model->ifft_plan);
    fftw_free(model->psi_k);
    fftw_free(model->q_k);
    g_free(model->psi);
    g_free(model->q);
    g_free(model->u);
    g_free(model->v);
    g_free(model->topography);
    g_free(model);
}

// Numerical solver (simplified Runge-Kutta 4)
void qg_step_rk4(QGModel *model, QGConfig *config) {
    int N = model->grid_size;
    int L = model->layers;
    double dx = 1.0 / N; // Grid spacing
    
    // Temporary arrays for RK4
    double *k1 = g_malloc(N * N * L * sizeof(double));
    double *k2 = g_malloc(N * N * L * sizeof(double));
    double *k3 = g_malloc(N * N * L * sizeof(double));
    double *k4 = g_malloc(N * N * L * sizeof(double));
    double *q_temp = g_malloc(N * N * L * sizeof(double));
    
    // Compute derivatives (PV advection, beta effect, and forcing)
    for (int l = 0; l < L; l++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int idx = l * N * N + i * N + j;
                int ip1 = (i + 1) % N, im1 = (i - 1 + N) % N;
                int jp1 = (j + 1) % N, jm1 = (j - 1 + N) % N;
                
                // Compute velocities from streamfunction
                model->u[idx] = -(model->psi[l * N * N + i * N + jp1] - model->psi[l * N * N + i * N + jm1]) / (2.0 * dx);
                model->v[idx] = (model->psi[l * N * N + ip1 * N + j] - model->psi[l * N * N + im1 * N + j]) / (2.0 * dx);
                
                // Advection and beta effect
                double dq_dx = (model->q[l * N * N + ip1 * N + j] - model->q[l * N * N + im1 * N + j]) / (2.0 * dx);
                double dq_dy = (model->q[l * N * N + i * N + jp1] - model->q[l * N * N + i * N + jm1]) / (2.0 * dx);
                k1[idx] = -model->u[idx] * dq_dx - model->v[idx] * dq_dy - config->beta * model->v[idx];
                
                // Add wind forcing (top layer) and bottom friction
                if (l == 0) k1[idx] += config->wind_forcing;
                if (l == L - 1) k1[idx] -= 0.1 * model->q[idx]; // Simplified friction
            }
        }
    }
    
    // Simplified RK4 (using only k1 for brevity)
    memcpy(q_temp, model->q, N * N * L * sizeof(double));
    for (int i = 0; i < N * N * L; i++) {
        model->q[i] += config->dt * k1[i];
    }
    
    // Update streamfunction (simplified PV inversion)
    for (int l = 0; l < L; l++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int idx = l * N * N + i * N + j;
                model->psi[idx] = model->q[idx] / (config->f0 * config->f0 / config->H); // Simplified inversion
            }
        }
    }
    
    g_free(k1);
    g_free(k2);
    g_free(k3);
    g_free(k4);
    g_free(q_temp);
}

// Drawing callback
static void draw_canvas(GtkDrawingArea *area, cairo_t *cr, int width, int height, gpointer user_data) {
    AppData *app = (AppData *)user_data;
    
    // Scale to fit
    double scale = MIN(width, height) / (double)app->model->grid_size;
    cairo_scale(cr, scale, scale);
    
    // Normalize psi for visualization
    double max_psi = 1e-10;
    for (int i = 0; i < app->model->grid_size * app->model->grid_size; i++) {
        max_psi = fmax(max_psi, fabs(app->model->psi[i]));
    }
    
    // Draw streamfunction (top layer)
    for (int i = 0; i < app->model->grid_size; i++) {
        for (int j = 0; j < app->model->grid_size; j++) {
            int idx = i * app->model->grid_size + j;
            double val = app->model->psi[idx] / (max_psi + 1e-10);
            double r = fmax(0, val);
            double b = fmax(0, -val);
            cairo_set_source_rgb(cr, r, 0, b);
            cairo_rectangle(cr, j, i, 1, 1);
            cairo_fill(cr);
        }
    }
}

// Simulation step
static gboolean simulation_step(gpointer user_data) {
    AppData *app = (AppData *)user_data;
    if (!app->running) return TRUE;
    
    qg_step_rk4(app->model, &app->config);
    gtk_widget_queue_draw(app->canvas);
    
    return TRUE;
}

// Start/Stop simulation
static void toggle_simulation(GtkButton *button, AppData *app) {
    app->running = !app->running;
    gtk_button_set_label(button, app->running ? "Stop" : "Start");
}

// Update configuration
static void update_config(AppData *app) {
    app->config.layers = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(app->layers_spin));
    
    GtkStringObject *selected;
    
    selected = GTK_STRING_OBJECT(gtk_drop_down_get_selected_item(GTK_DROP_DOWN(app->geometry_dropdown)));
    strcpy(app->config.geometry, gtk_string_object_get_string(selected));
    
    selected = GTK_STRING_OBJECT(gtk_drop_down_get_selected_item(GTK_DROP_DOWN(app->solver_dropdown)));
    strcpy(app->config.solver, gtk_string_object_get_string(selected));
    
    selected = GTK_STRING_OBJECT(gtk_drop_down_get_selected_item(GTK_DROP_DOWN(app->boundary_dropdown)));
    strcpy(app->config.boundary, gtk_string_object_get_string(selected));
    
    const char *beta_text = gtk_entry_buffer_get_text(gtk_entry_get_buffer(GTK_ENTRY(app->beta_entry)));
    app->config.beta = g_ascii_strtod(beta_text, NULL);
    
    const char *dt_text = gtk_entry_buffer_get_text(gtk_entry_get_buffer(GTK_ENTRY(app->dt_entry)));
    app->config.dt = g_ascii_strtod(dt_text, NULL);
    
    app->config.wind_forcing = 1e-6; // Example value
    
    // Reinitialize model if layers changed
    if (app->model->layers != app->config.layers) {
        qg_model_free(app->model);
        app->model = qg_model_init(app->config.layers, GRID_SIZE);
    }
}

// Create GUI
static void activate(GtkApplication *app, gpointer user_data) {
    AppData *app_data = g_malloc(sizeof(AppData));
    app_data->model = qg_model_init(2, GRID_SIZE);
    app_data->running = FALSE;
    app_data->config.layers = 2;
    app_data->config.grid_size = GRID_SIZE;
    app_data->config.dt = DEFAULT_DT;
    app_data->config.beta = DEFAULT_BETA;
    app_data->config.f0 = DEFAULT_F0;
    app_data->config.g = DEFAULT_G;
    app_data->config.H = DEFAULT_H;
    strcpy(app_data->config.geometry, "beta-plane");
    strcpy(app_data->config.solver, "runge-kutta");
    strcpy(app_data->config.boundary, "periodic");
    app_data->config.wind_forcing = 1e-6;
    
    // Window
    app_data->window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(app_data->window), "QG Model");
    gtk_window_set_default_size(GTK_WINDOW(app_data->window), 360, 640); // Mobile-like
    
    // Main box
    GtkWidget *main_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_start(main_box, 10);
    gtk_widget_set_margin_end(main_box, 10);
    gtk_widget_set_margin_top(main_box, 10);
    gtk_widget_set_margin_bottom(main_box, 10);
    gtk_window_set_child(GTK_WINDOW(app_data->window), main_box);
    
    // Configuration box
    app_data->config_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_box_append(GTK_BOX(main_box), app_data->config_box);
    
    // Layers
    GtkWidget *layers_label = gtk_label_new("Layers:");
    gtk_box_append(GTK_BOX(app_data->config_box), layers_label);
    app_data->layers_spin = gtk_spin_button_new_with_range(2, 10, 1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(app_data->layers_spin), 2);
    gtk_box_append(GTK_BOX(app_data->config_box), app_data->layers_spin);
    
    // Geometry
    GtkWidget *geometry_label = gtk_label_new("Geometry:");
    gtk_box_append(GTK_BOX(app_data->config_box), geometry_label);
    GtkStringList *geometry_list = gtk_string_list_new(NULL);
    gtk_string_list_append(geometry_list, "beta-plane");
    gtk_string_list_append(geometry_list, "spherical");
    app_data->geometry_dropdown = gtk_drop_down_new(G_LIST_MODEL(geometry_list), NULL);
    gtk_drop_down_set_selected(GTK_DROP_DOWN(app_data->geometry_dropdown), 0);
    gtk_box_append(GTK_BOX(app_data->config_box), app_data->geometry_dropdown);
    
    // Solver
    GtkWidget *solver_label = gtk_label_new("Solver:");
    gtk_box_append(GTK_BOX(app_data->config_box), solver_label);
    GtkStringList *solver_list = gtk_string_list_new(NULL);
    gtk_string_list_append(solver_list, "runge-kutta");
    gtk_string_list_append(solver_list, "leapfrog");
    gtk_string_list_append(solver_list, "semi-implicit");
    app_data->solver_dropdown = gtk_drop_down_new(G_LIST_MODEL(solver_list), NULL);
    gtk_drop_down_set_selected(GTK_DROP_DOWN(app_data->solver_dropdown), 0);
    gtk_box_append(GTK_BOX(app_data->config_box), app_data->solver_dropdown);
    
    // Boundary
    GtkWidget *boundary_label = gtk_label_new("Boundary:");
    gtk_box_append(GTK_BOX(app_data->config_box), boundary_label);
    GtkStringList *boundary_list = gtk_string_list_new(NULL);
    gtk_string_list_append(boundary_list, "periodic");
    gtk_string_list_append(boundary_list, "rigid");
    gtk_string_list_append(boundary_list, "open");
    app_data->boundary_dropdown = gtk_drop_down_new(G_LIST_MODEL(boundary_list), NULL);
    gtk_drop_down_set_selected(GTK_DROP_DOWN(app_data->boundary_dropdown), 0);
    gtk_box_append(GTK_BOX(app_data->config_box), app_data->boundary_dropdown);
    
    // Beta
    GtkWidget *beta_label = gtk_label_new("Beta:");
    gtk_box_append(GTK_BOX(app_data->config_box), beta_label);
    app_data->beta_entry = gtk_entry_new();
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(app_data->beta_entry)), "1.6e-11", -1);
    gtk_box_append(GTK_BOX(app_data->config_box), app_data->beta_entry);
    
    // Time step
    GtkWidget *dt_label = gtk_label_new("Time Step:");
    gtk_box_append(GTK_BOX(app_data->config_box), dt_label);
    app_data->dt_entry = gtk_entry_new();
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(app_data->dt_entry)), "0.01", -1);
    gtk_box_append(GTK_BOX(app_data->config_box), app_data->dt_entry);
    
    // Start/Stop button
    GtkWidget *start_button = gtk_button_new_with_label("Start");
    g_signal_connect(start_button, "clicked", G_CALLBACK(toggle_simulation), app_data);
    gtk_box_append(GTK_BOX(app_data->config_box), start_button);
    
    // Canvas
    app_data->canvas = gtk_drawing_area_new();
    gtk_widget_set_size_request(app_data->canvas, 300, 300);
    gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(app_data->canvas), draw_canvas, app_data, NULL);
    gtk_box_append(GTK_BOX(main_box), app_data->canvas);
    
    // Update configuration on change
    g_signal_connect_swapped(app_data->layers_spin, "value-changed", G_CALLBACK(update_config), app_data);
    g_signal_connect_swapped(app_data->geometry_dropdown, "notify::selected", G_CALLBACK(update_config), app_data);
    g_signal_connect_swapped(app_data->solver_dropdown, "notify::selected", G_CALLBACK(update_config), app_data);
    g_signal_connect_swapped(app_data->boundary_dropdown, "notify::selected", G_CALLBACK(update_config), app_data);
    
    // Simulation loop
    g_timeout_add(100, simulation_step, app_data);
    
    gtk_window_present(GTK_WINDOW(app_data->window));
}

int main(int argc, char *argv[]) {
    GtkApplication *app = gtk_application_new("org.xai.qgmodel", G_APPLICATION_DEFAULT_FLAGS);
    g_signal_connect(app, "activate", G_CALLBACK(activate), NULL);
    int status = g_application_run(G_APPLICATION(app), argc, argv);
    g_object_unref(app);
    return status;
}