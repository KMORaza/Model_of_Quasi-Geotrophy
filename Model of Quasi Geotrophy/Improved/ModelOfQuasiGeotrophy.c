#include <gtk/gtk.h>
#include <math.h>
#include <fftw3.h>
#include <string.h>

#ifdef MIN
#undef MIN
#endif
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#ifdef MAX
#undef MAX
#endif
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define GRID_SIZE 64
#define DEFAULT_DT 3600.0 // 1 hour
#define DEFAULT_BETA 2e-11 
#define DEFAULT_F0 1e-4
#define DEFAULT_G 9.81
#define DEFAULT_H 2500.0
#define SPONGE_WIDTH 5
#define SPONGE_STRENGTH 0.1
#define DIFFUSION_COEFF 5e-8 
#define DOMAIN_LENGTH 1e6 // 1000 km

// Model configuration structure
typedef struct {
    int layers;
    int grid_size;
    double dt;
    double beta;
    double f0;
    double g;
    double H;
    char geometry[16];
    char solver[16];
    char boundary[16];
    double wind_forcing;
    double topo_scale;
} QGConfig;

// Simulation data
typedef struct {
    int layers;
    int grid_size;
    double *psi;
    double *q;
    double *q_prev;
    double *u, *v;
    double *topography;
    fftw_complex *psi_k, *q_k;
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
    GtkWidget *wind_entry;
    GtkWidget *topo_spin;
    GtkWidget *start_button;
    GtkWidget *reset_button;
    GtkWidget *field_dropdown;
    GtkWidget *vis_layer_spin;
    GtkWidget *energy_label;
    GtkWidget *rossby_label;
    GtkWidget *canvas; // Added canvas member
    QGModel *model;
    QGConfig config;
    gboolean running;
    int vis_layer;
    char vis_field[16];
} AppData;

// Initialize model
QGModel *qg_model_init(int layers, int grid_size) {
    QGModel *model = g_malloc(sizeof(QGModel));
    model->layers = layers;
    model->grid_size = grid_size;
    
    // Allocate arrays
    model->psi = g_malloc(grid_size * grid_size * layers * sizeof(double));
    model->q = g_malloc(grid_size * grid_size * layers * sizeof(double));
    model->q_prev = g_malloc(grid_size * grid_size * layers * sizeof(double));
    model->u = g_malloc(grid_size * grid_size * layers * sizeof(double));
    model->v = g_malloc(grid_size * grid_size * layers * sizeof(double));
    model->topography = g_malloc(grid_size * grid_size * sizeof(double));
    model->psi_k = fftw_malloc(grid_size * grid_size * layers * sizeof(fftw_complex));
    model->q_k = fftw_malloc(grid_size * grid_size * layers * sizeof(fftw_complex));
    
    // Initialize FFTW plans
    model->fft_plan = fftw_plan_dft_r2c_2d(grid_size, grid_size, model->psi, model->psi_k, FFTW_MEASURE);
    model->ifft_plan = fftw_plan_dft_c2r_2d(grid_size, grid_size, model->psi_k, model->psi, FFTW_MEASURE);
    
    // Initialize fields
    memset(model->psi, 0, grid_size * grid_size * layers * sizeof(double));
    memset(model->q, 0, grid_size * grid_size * layers * sizeof(double));
    memset(model->q_prev, 0, grid_size * grid_size * layers * sizeof(double));
    memset(model->topography, 0, grid_size * grid_size * sizeof(double));
    
    // Initialize with a zonal jet plus stronger perturbations
    for (int l = 0; l < layers; l++) {
        double amplitude = (l == 0) ? 1e5 : 5e4;
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                int idx = l * grid_size * grid_size + i * grid_size + j;
                double y = (j - grid_size / 2.0) * DOMAIN_LENGTH / grid_size;
                model->psi[idx] = amplitude * sin(2.0 * M_PI * y / DOMAIN_LENGTH) * exp(-pow(y / (DOMAIN_LENGTH / 4.0), 2));
                model->psi[idx] += 5e3 * sin(4.0 * M_PI * i / grid_size) * sin(4.0 * M_PI * j / grid_size);
                model->q[idx] = 0.0;
                model->q_prev[idx] = model->q[idx];
            }
        }
    }
    
    // Initialize topography (ridge)
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            int idx = i * grid_size + j;
            double y = (j - grid_size / 2.0) * DOMAIN_LENGTH / grid_size;
            model->topography[idx] = 500.0 * exp(-pow(y / (DOMAIN_LENGTH / 8.0), 2));
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
    g_free(model->q_prev);
    g_free(model->u);
    g_free(model->v);
    g_free(model->topography);
    g_free(model);
}

// Spectral PV inversion
void invert_pv(QGModel *model, QGConfig *config) {
    int N = model->grid_size;
    int L = model->layers;
    double dx = DOMAIN_LENGTH / N;
    
    // FFT of q
    for (int l = 0; l < L; l++) {
        double *q_layer = model->q + l * N * N;
        fftw_execute_dft_r2c(model->fft_plan, q_layer, model->q_k + l * N * (N / 2 + 1));
    }
    
    // 2-layer PV inversion
    double f0 = config->f0;
    double H = config->H;
    double stretch = f0 * f0 / (config->g * H);
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N / 2 + 1; j++) {
            int idx = i * (N / 2 + 1) + j;
            double kx = 2.0 * M_PI * (i < N / 2 ? i : i - N) / (N * dx);
            double ky = 2.0 * M_PI * j / (N * dx);
            double k2 = kx * kx + ky * ky;
            
            fftw_complex q_k[L], psi_k[L];
            for (int l = 0; l < L; l++) {
                q_k[l][0] = model->q_k[l * N * (N / 2 + 1) + idx][0];
                q_k[l][1] = model->q_k[l * N * (N / 2 + 1) + idx][1];
            }
            
            if (L == 2) {
                double s = stretch;
                double denom = (k2 + s) * (k2 + s) - s * s;
                if (denom < 1e-10) denom = 1e-10;
                
                psi_k[0][0] = -((k2 + s) * q_k[0][0] - s * q_k[1][0]) / denom;
                psi_k[0][1] = -((k2 + s) * q_k[0][1] - s * q_k[1][1]) / denom;
                psi_k[1][0] = -(-s * q_k[0][0] + (k2 + s) * q_k[1][0]) / denom;
                psi_k[1][1] = -(-s * q_k[0][1] + (k2 + s) * q_k[1][1]) / denom;
            } else {
                for (int l = 0; l < L; l++) {
                    double denom = k2 + stretch * (l == L-1 ? 1.0 : 1.5);
                    if (denom < 1e-10) denom = 1e-10;
                    psi_k[l][0] = -q_k[l][0] / denom;
                    psi_k[l][1] = -q_k[l][1] / denom;
                }
            }
            
            for (int l = 0; l < L; l++) {
                model->psi_k[l * N * (N / 2 + 1) + idx][0] = psi_k[l][0];
                model->psi_k[l * N * (N / 2 + 1) + idx][1] = psi_k[l][1];
            }
        }
    }
    
    // Inverse FFT
    for (int l = 0; l < L; l++) {
        double *psi_layer = model->psi + l * N * N;
        fftw_execute_dft_c2r(model->ifft_plan, model->psi_k + l * N * (N / 2 + 1), psi_layer);
        for (int i = 0; i < N * N; i++) {
            psi_layer[i] /= (N * N);
        }
    }
}

// Compute derivatives
void compute_derivatives(QGModel *model, QGConfig *config, double *dq_dt) {
    int N = model->grid_size;
    int L = model->layers;
    double dx = DOMAIN_LENGTH / N;
    
    double *laplacian = g_malloc(N * N * L * sizeof(double));
    
    for (int l = 0; l < L; l++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int idx = l * N * N + i * N + j;
                
                int ip1 = (strcmp(config->boundary, "periodic") == 0) ? (i + 1) % N : MIN(i + 1, N - 1);
                int im1 = (strcmp(config->boundary, "periodic") == 0) ? (i - 1 + N) % N : MAX(i - 1, 0);
                int jp1 = (strcmp(config->boundary, "periodic") == 0) ? (j + 1) % N : MIN(j + 1, N - 1);
                int jm1 = (strcmp(config->boundary, "periodic") == 0) ? (j - 1 + N) % N : MAX(j - 1, 0);
                
                if (strcmp(config->boundary, "rigid") == 0 && (i == 0 || i == N-1 || j == 0 || j == N-1)) {
                    model->u[idx] = 0.0;
                    model->v[idx] = 0.0;
                    dq_dt[idx] = 0.0;
                    laplacian[idx] = 0.0;
                    continue;
                }
                
                // Velocities
                model->u[idx] = -(model->psi[l * N * N + i * N + jp1] - model->psi[l * N * N + i * N + jm1]) / (2.0 * dx);
                model->v[idx] = (model->psi[l * N * N + ip1 * N + j] - model->psi[l * N * N + im1 * N + j]) / (2.0 * dx);
                
                // Advection
                double dq_dx = (model->q[l * N * N + ip1 * N + j] - model->q[l * N * N + im1 * N + j]) / (2.0 * dx);
                double dq_dy = (model->q[l * N * N + i * N + jp1] - model->q[l * N * N + i * N + jm1]) / (2.0 * dx);
                dq_dt[idx] = -model->u[idx] * dq_dx - model->v[idx] * dq_dy;
                
                // Geometry
                if (strcmp(config->geometry, "beta-plane") == 0) {
                    dq_dt[idx] -= config->beta * model->v[idx];
                } else if (strcmp(config->geometry, "spherical") == 0) {
                    double lat = ((double)j / N - 0.5) * 2.0 * M_PI / 3.0; // ±60°
                    double beta = 2.0 * 7.2921e-5 * cos(lat) / 6.371e6;
                    dq_dt[idx] -= beta * model->v[idx];
                }
                
                // Double-gyre wind forcing
                if (l == 0) {
                    double y = (double)j / N;
                    double tau = config->wind_forcing * sin(2.0 * M_PI * y);
                    dq_dt[idx] += tau / (config->H * 1000.0);
                }
                
                // Friction
                if (l == L - 1) {
                    dq_dt[idx] -= 1e-7 * model->q[idx];
                }
                
                // Topography
                if (l == L - 1) {
                    double h = config->topo_scale * model->topography[i * N + j];
                    double dh_dx = config->topo_scale * (model->topography[i * N + jp1] - model->topography[i * N + jm1]) / (2.0 * dx);
                    double dh_dy = config->topo_scale * (model->topography[ip1 * N + j] - model->topography[im1 * N + j]) / (2.0 * dx);
                    dq_dt[idx] -= (config->f0 / config->H) * (model->u[idx] * dh_dx + model->v[idx] * dh_dy);
                }
                
                // Laplacian for hyperdiffusion
                laplacian[idx] = (model->q[l * N * N + ip1 * N + j] + model->q[l * N * N + im1 * N + j] +
                                 model->q[l * N * N + i * N + jp1] + model->q[l * N * N + i * N + jm1] -
                                 4.0 * model->q[idx]) / (dx * dx);
            }
        }
    }
    
    // Hyperdiffusion
    for (int l = 0; l < L; l++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int idx = l * N * N + i * N + j;
                if (strcmp(config->boundary, "rigid") == 0 && (i == 0 || i == N-1 || j == 0 || j == N-1)) continue;
                int ip1 = (strcmp(config->boundary, "periodic") == 0) ? (i + 1) % N : MIN(i + 1, N - 1);
                int im1 = (strcmp(config->boundary, "periodic") == 0) ? (i - 1 + N) % N : MAX(i - 1, 0);
                int jp1 = (strcmp(config->boundary, "periodic") == 0) ? (j + 1) % N : MIN(j + 1, N - 1);
                int jm1 = (strcmp(config->boundary, "periodic") == 0) ? (j - 1 + N) % N : MAX(j - 1, 0);
                double lap2 = (laplacian[l * N * N + ip1 * N + j] + laplacian[l * N * N + im1 * N + j] +
                               laplacian[l * N * N + i * N + jp1] + laplacian[l * N * N + i * N + jm1] -
                               4.0 * laplacian[idx]) / (dx * dx);
                dq_dt[idx] -= DIFFUSION_COEFF * lap2;
                
                // Sponge layer
                if (strcmp(config->boundary, "open") == 0) {
                    double sponge = 0.0;
                    if (i < SPONGE_WIDTH) sponge = SPONGE_STRENGTH * (SPONGE_WIDTH - i) / SPONGE_WIDTH;
                    if (i >= N - SPONGE_WIDTH) sponge = SPONGE_STRENGTH * (i - (N - SPONGE_WIDTH)) / SPONGE_WIDTH;
                    if (j < SPONGE_WIDTH) sponge = SPONGE_STRENGTH * (SPONGE_WIDTH - j) / SPONGE_WIDTH;
                    if (j >= N - SPONGE_WIDTH) sponge = SPONGE_STRENGTH * (j - (N - SPONGE_WIDTH)) / SPONGE_WIDTH;
                    dq_dt[idx] -= sponge * model->q[idx];
                }
            }
        }
    }
    
    g_free(laplacian);
}

// Runge-Kutta 4 solver
void qg_step_rk4(QGModel *model, QGConfig *config) {
    int N = model->grid_size;
    int L = model->layers;
    
    double *k1 = g_malloc(N * N * L * sizeof(double));
    double *k2 = g_malloc(N * N * L * sizeof(double));
    double *k3 = g_malloc(N * N * L * sizeof(double));
    double *k4 = g_malloc(N * N * L * sizeof(double));
    double *q_temp = g_malloc(N * N * L * sizeof(double));
    
    compute_derivatives(model, config, k1);
    for (int i = 0; i < N * N * L; i++) {
        q_temp[i] = model->q[i] + 0.5 * config->dt * k1[i];
    }
    memcpy(model->q, q_temp, N * N * L * sizeof(double));
    invert_pv(model, config);
    compute_derivatives(model, config, k2);
    
    for (int i = 0; i < N * N * L; i++) {
        q_temp[i] = model->q[i] + 0.5 * config->dt * k2[i];
    }
    memcpy(model->q, q_temp, N * N * L * sizeof(double));
    invert_pv(model, config);
    compute_derivatives(model, config, k3);
    
    for (int i = 0; i < N * N * L; i++) {
        q_temp[i] = model->q[i] + config->dt * k3[i];
    }
    memcpy(model->q, q_temp, N * N * L * sizeof(double));
    invert_pv(model, config);
    compute_derivatives(model, config, k4);
    
    for (int i = 0; i < N * N * L; i++) {
        model->q[i] += (config->dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    
    invert_pv(model, config);
    
    g_free(k1);
    g_free(k2);
    g_free(k3);
    g_free(k4);
    g_free(q_temp);
}

// Leapfrog solver
void qg_step_leapfrog(QGModel *model, QGConfig *config) {
    int N = model->grid_size;
    int L = model->layers;
    
    double *dq_dt = g_malloc(N * N * L * sizeof(double));
    
    compute_derivatives(model, config, dq_dt);
    
    for (int i = 0; i < N * N * L; i++) {
        double q_new = model->q_prev[i] + 2.0 * config->dt * dq_dt[i];
        model->q_prev[i] = model->q[i];
        model->q[i] = q_new;
    }
    
    invert_pv(model, config);
    
    g_free(dq_dt);
}

// Semi-Implicit solver
void qg_step_semi_implicit(QGModel *model, QGConfig *config) {
    int N = model->grid_size;
    int L = model->layers;
    
    double *dq_dt = g_malloc(N * N * L * sizeof(double));
    double *q_temp = g_malloc(N * N * L * sizeof(double));
    
    compute_derivatives(model, config, dq_dt);
    for (int i = 0; i < N * N * L; i++) {
        q_temp[i] = model->q[i] + config->dt * dq_dt[i];
    }
    memcpy(model->q, q_temp, N * N * L * sizeof(double));
    invert_pv(model, config);
    
    for (int l = 0; l < L; l++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int idx = l * N * N + i * N + j;
                double friction = (l == L - 1) ? 1e-7 : 0.0;
                double beta_term = 0.0;
                if (strcmp(config->geometry, "beta-plane") == 0) {
                    beta_term = config->beta * model->v[idx];
                } else if (strcmp(config->geometry, "spherical") == 0) {
                    double lat = ((double)j / N - 0.5) * 2.0 * M_PI / 3.0;
                    double beta = 2.0 * 7.2921e-5 * cos(lat) / 6.371e6;
                    beta_term = beta * model->v[idx];
                }
                double denom = 1.0 + config->dt * friction;
                model->q[idx] = (q_temp[idx] - config->dt * beta_term) / denom;
            }
        }
    }
    
    invert_pv(model, config);
    
    g_free(dq_dt);
    g_free(q_temp);
}

// Compute total energy
static double compute_energy(AppData *app) {
    QGModel *model = app->model;
    int N = model->grid_size;
    int L = model->layers;
    double energy = 0.0;
    
    for (int l = 0; l < L; l++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int idx = l * N * N + i * N + j;
                energy += 0.5 * (model->u[idx] * model->u[idx] + model->v[idx] * model->v[idx]);
            }
        }
    }
    
    return energy / (N * N * L);
}

// Compute representative Rossby wave speed
static double compute_rossby_speed(AppData *app) {
    QGConfig *config = &app->config;
    int N = app->model->grid_size;
    double dx = DOMAIN_LENGTH / N;
    double k = 2.0 * M_PI / (N * dx); // Typical wavenumber
    double k2 = k * k;
    
    double beta;
    if (strcmp(config->geometry, "beta-plane") == 0) {
        beta = config->beta;
    } else {
        // Average beta at mid-domain (lat = 0)
        double lat = 0.0;
        beta = 2.0 * 7.2921e-5 * cos(lat) / 6.371e6;
    }
    
    return -beta / k2; // Rossby wave speed (m/s)
}

// Drawing callback
static void draw_canvas(GtkDrawingArea *area, cairo_t *cr, int width, int height, gpointer user_data) {
    AppData *app = (AppData *)user_data;
    
    double scale = MIN(width, height) / (double)app->model->grid_size;
    cairo_scale(cr, scale, scale);
    
    // Select field
    double *field = NULL;
    if (strcmp(app->vis_field, "psi") == 0) {
        field = app->model->psi + app->vis_layer * app->model->grid_size * app->model->grid_size;
    } else if (strcmp(app->vis_field, "q") == 0) {
        field = app->model->q + app->vis_layer * app->model->grid_size * app->model->grid_size;
    } else if (strcmp(app->vis_field, "velocity") == 0) {
        field = g_malloc(app->model->grid_size * app->model->grid_size * sizeof(double));
        for (int i = 0; i < app->model->grid_size; i++) {
            for (int j = 0; j < app->model->grid_size; j++) {
                int idx = i * app->model->grid_size + j;
                int model_idx = app->vis_layer * app->model->grid_size * app->model->grid_size + idx;
                field[idx] = sqrt(app->model->u[model_idx] * app->model->u[model_idx] + 
                                 app->model->v[model_idx] * app->model->v[model_idx]);
            }
        }
    }
    
    // Normalize
    double min_val = field[0], max_val = field[0];
    for (int i = 0; i < app->model->grid_size * app->model->grid_size; i++) {
        min_val = fmin(min_val, field[i]);
        max_val = fmax(max_val, field[i]);
    }
    double range = (max_val - min_val) + 1e-10; // More sensitive normalization
    
    // Draw with enhanced colormap
    for (int i = 0; i < app->model->grid_size; i++) {
        for (int j = 0; j < app->model->grid_size; j++) {
            int idx = i * app->model->grid_size + j;
            double val = 2.0 * (field[idx] - min_val) / range - 1.0; // [-1, 1]
            double r = val > 0 ? 0.8 + 0.2 * val : 0.2 * (1.0 + val);
            double b = val < 0 ? 0.8 - 0.2 * val : 0.2 * (1.0 - val);
            double g = 0.2 + 0.6 * (1.0 - fabs(val));
            cairo_set_source_rgb(cr, r, g, b);
            cairo_rectangle(cr, j, i, 1, 1);
            cairo_fill(cr);
        }
    }
    
    if (strcmp(app->vis_field, "velocity") == 0) {
        g_free(field);
    }
}

// Simulation step
static gboolean simulation_step(gpointer user_data) {
    AppData *app = (AppData *)user_data;
    if (!app->running) return TRUE;
    
    if (strcmp(app->config.solver, "runge-kutta") == 0) {
        qg_step_rk4(app->model, &app->config);
    } else if (strcmp(app->config.solver, "leapfrog") == 0) {
        qg_step_leapfrog(app->model, &app->config);
    } else if (strcmp(app->config.solver, "semi-implicit") == 0) {
        qg_step_semi_implicit(app->model, &app->config);
    }
    
    // Update diagnostics
    double energy = compute_energy(app);
    char energy_text[64];
    snprintf(energy_text, sizeof(energy_text), "Energy: %.2e", energy);
    gtk_label_set_text(GTK_LABEL(app->energy_label), energy_text);
    
    double rossby_speed = compute_rossby_speed(app);
    char rossby_text[64];
    snprintf(rossby_text, sizeof(rossby_text), "Rossby Speed: %.2e m/s", rossby_speed);
    gtk_label_set_text(GTK_LABEL(app->rossby_label), rossby_text);
    
    gtk_widget_queue_draw(app->canvas);
    
    return TRUE;
}

// Start/Stop simulation
static void toggle_simulation(GtkButton *button, AppData *app) {
    app->running = !app->running;
    gtk_button_set_label(button, app->running ? "Stop" : "Start");
}

// Reset simulation
static void reset_simulation(GtkButton *button, AppData *app) {
    app->running = FALSE;
    gtk_button_set_label(GTK_BUTTON(app->start_button), "Start");
    qg_model_free(app->model);
    app->model = qg_model_init(app->config.layers, GRID_SIZE);
    invert_pv(app->model, &app->config);
    gtk_widget_queue_draw(app->canvas);
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
    
    selected = GTK_STRING_OBJECT(gtk_drop_down_get_selected_item(GTK_DROP_DOWN(app->field_dropdown)));
    strcpy(app->vis_field, gtk_string_object_get_string(selected));
    
    const char *beta_text = gtk_entry_buffer_get_text(gtk_entry_get_buffer(GTK_ENTRY(app->beta_entry)));
    app->config.beta = g_ascii_strtod(beta_text, NULL);
    
    const char *dt_text = gtk_entry_buffer_get_text(gtk_entry_get_buffer(GTK_ENTRY(app->dt_entry)));
    app->config.dt = g_ascii_strtod(dt_text, NULL);
    
    const char *wind_text = gtk_entry_buffer_get_text(gtk_entry_get_buffer(GTK_ENTRY(app->wind_entry)));
    app->config.wind_forcing = g_ascii_strtod(wind_text, NULL);
    
    app->config.topo_scale = gtk_spin_button_get_value(GTK_SPIN_BUTTON(app->topo_spin));
    
    app->vis_layer = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(app->vis_layer_spin));
    if (app->vis_layer >= app->config.layers) {
        app->vis_layer = app->config.layers - 1;
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(app->vis_layer_spin), app->vis_layer);
    }
    
    if (app->model->layers != app->config.layers) {
        qg_model_free(app->model);
        app->model = qg_model_init(app->config.layers, GRID_SIZE);
        invert_pv(app->model, &app->config);
    }
    
    gtk_widget_queue_draw(app->canvas);
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
    strcpy(app_data->vis_field, "psi");
    app_data->vis_layer = 0;
    app_data->config.wind_forcing = 1e-7;
    app_data->config.topo_scale = 1.0;
    
    // Initialize q
    invert_pv(app_data->model, &app_data->config);
    
    // Window
    app_data->window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(app_data->window), "QG Ocean Model");
    gtk_window_set_default_size(GTK_WINDOW(app_data->window), 360, 740);
    
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
    app_data->layers_spin = gtk_spin_button_new_with_range(2, 4, 1);
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
    GtkWidget *beta_label = gtk_label_new("Beta (m^-1 s^-1):");
    gtk_box_append(GTK_BOX(app_data->config_box), beta_label);
    app_data->beta_entry = gtk_entry_new();
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(app_data->beta_entry)), "2e-11", -1);
    gtk_box_append(GTK_BOX(app_data->config_box), app_data->beta_entry);
    
    // Time step
    GtkWidget *dt_label = gtk_label_new("Time Step (s):");
    gtk_box_append(GTK_BOX(app_data->config_box), dt_label);
    app_data->dt_entry = gtk_entry_new();
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(app_data->dt_entry)), "3600", -1);
    gtk_box_append(GTK_BOX(app_data->config_box), app_data->dt_entry);
    
    // Wind forcing
    GtkWidget *wind_label = gtk_label_new("Wind Stress (N/m^2):");
    gtk_box_append(GTK_BOX(app_data->config_box), wind_label);
    app_data->wind_entry = gtk_entry_new();
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(app_data->wind_entry)), "1e-7", -1);
    gtk_box_append(GTK_BOX(app_data->config_box), app_data->wind_entry);
    
    // Topography scale
    GtkWidget *topo_label = gtk_label_new("Topography Scale:");
    gtk_box_append(GTK_BOX(app_data->config_box), topo_label);
    app_data->topo_spin = gtk_spin_button_new_with_range(0.0, 2.0, 0.1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(app_data->topo_spin), 1.0);
    gtk_box_append(GTK_BOX(app_data->config_box), app_data->topo_spin);
    
    // Field selection
    GtkWidget *field_label = gtk_label_new("Visualize Field:");
    gtk_box_append(GTK_BOX(app_data->config_box), field_label);
    GtkStringList *field_list = gtk_string_list_new(NULL);
    gtk_string_list_append(field_list, "psi");
    gtk_string_list_append(field_list, "q");
    gtk_string_list_append(field_list, "velocity");
    app_data->field_dropdown = gtk_drop_down_new(G_LIST_MODEL(field_list), NULL);
    gtk_drop_down_set_selected(GTK_DROP_DOWN(app_data->field_dropdown), 0);
    gtk_box_append(GTK_BOX(app_data->config_box), app_data->field_dropdown);
    
    // Layer selection
    GtkWidget *vis_layer_label = gtk_label_new("Visualize Layer:");
    gtk_box_append(GTK_BOX(app_data->config_box), vis_layer_label);
    app_data->vis_layer_spin = gtk_spin_button_new_with_range(0, app_data->config.layers - 1, 1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(app_data->vis_layer_spin), 0);
    gtk_box_append(GTK_BOX(app_data->config_box), app_data->vis_layer_spin);
    
    // Start/Stop button
    app_data->start_button = gtk_button_new_with_label("Start");
    g_signal_connect(app_data->start_button, "clicked", G_CALLBACK(toggle_simulation), app_data);
    gtk_box_append(GTK_BOX(app_data->config_box), app_data->start_button);
    
    // Reset button
    app_data->reset_button = gtk_button_new_with_label("Reset");
    g_signal_connect(app_data->reset_button, "clicked", G_CALLBACK(reset_simulation), app_data);
    gtk_box_append(GTK_BOX(app_data->config_box), app_data->reset_button);
    
    // Energy display
    app_data->energy_label = gtk_label_new("Energy: 0.0");
    gtk_box_append(GTK_BOX(app_data->config_box), app_data->energy_label);
    
    // Rossby speed display
    app_data->rossby_label = gtk_label_new("Rossby Speed: 0.0 m/s");
    gtk_box_append(GTK_BOX(app_data->config_box), app_data->rossby_label);
    
    // Canvas
    app_data->canvas = gtk_drawing_area_new();
    gtk_widget_set_size_request(app_data->canvas, 300, 300);
    gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(app_data->canvas), draw_canvas, app_data, NULL);
    gtk_box_append(GTK_BOX(main_box), app_data->canvas);
    
    // Update configuration
    g_signal_connect_swapped(app_data->layers_spin, "value-changed", G_CALLBACK(update_config), app_data);
    g_signal_connect_swapped(app_data->geometry_dropdown, "notify::selected", G_CALLBACK(update_config), app_data);
    g_signal_connect_swapped(app_data->solver_dropdown, "notify::selected", G_CALLBACK(update_config), app_data);
    g_signal_connect_swapped(app_data->boundary_dropdown, "notify::selected", G_CALLBACK(update_config), app_data);
    g_signal_connect_swapped(app_data->field_dropdown, "notify::selected", G_CALLBACK(update_config), app_data);
    g_signal_connect_swapped(app_data->vis_layer_spin, "value-changed", G_CALLBACK(update_config), app_data);
    g_signal_connect_swapped(app_data->wind_entry, "changed", G_CALLBACK(update_config), app_data);
    g_signal_connect_swapped(app_data->topo_spin, "value-changed", G_CALLBACK(update_config), app_data);
    
    // Simulation loop
    g_timeout_add(20, simulation_step, app_data);
    
    gtk_window_present(GTK_WINDOW(app_data->window));
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    GtkApplication *app = gtk_application_new("org.xai.qgmodel", G_APPLICATION_DEFAULT_FLAGS);
    g_signal_connect(app, "activate", G_CALLBACK(activate), NULL);
    int status = g_application_run(G_APPLICATION(app), argc, argv);
    g_object_unref(app);
    return status;
}