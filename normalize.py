import sys
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from scipy.interpolate import splrep, splev
from astropy.io import fits

'''
Initially written for mercator spectra only by Julia Bodensteiner/Sanjay Sekaran
Works for any spectra now
'''

def write_hermes(infile, flux, outfile, wave=None):
    hdul_infile = fits.open(infile)
    hdul_new = fits.HDUList()
    primheader = hdul_infile[0].header.copy()   
    if 'CRVAL1' not in primheader:
        # >>>>> unevenly spaced - fix this <<<<<<<<
        primheader.set('CRVAL1', wave[0])  # Starting wl at CRPIX1
        primheader.set('CDELT1', wave[1]-wave[0])  # Stepwidth of wl
        primheader.set('CRPIX1', 1)  # Reference Pixel
        print("Added keywords to header")    
    
    hdul_new.append(fits.PrimaryHDU(data=flux, header=primheader))
    hdul_new.writeto(outfile)    
    print("Data written to %s" % outfile)






def read_HERMES(infile):
    print("%s: Input file is a HERMES file." % infile)
    header = fits.getheader(infile)
    # for files with standard wavelegth array
    #if ((header['CTYPE1'] == 'WAVELENGTH') or (header['CTYPE1'] == 'AWAV')):
    flux = fits.getdata(infile)
    crval = header['CRVAL1']
    cdelt = header['CDELT1']
    naxis1 = header['NAXIS1']
    wave = crval + np.arange(0, naxis1) * cdelt


    # for files that are given in logarithmic wl array
    #if (header['CTYPE1'] == 'log(wavelength)'):
    #    flux = fits.getdata(infile)
    #    crval = header['CRVAL1']
    #    cdelt = header['CDELT1']
    #    naxis1 = header['NAXIS1']
    #    wave = np.exp(crval + np.arange(0, naxis1)*cdelt)

    #else:
    #    print("Could not read in HERMES fits file - unknown file type.")
    #    sys.exit()

    return wave, flux


class PointBrowser(object):
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """

    def __init__(self, fig, ax1, ax2, w, f, filepath, files, file_number,
                 bounds):
        self.number_of_files = len(w)
        self.files = files

#        plot_title = self.files[file_number].split('/')[2].split('_202')[0].replace('_', ' ')
#        ax1.set_title(plot_title + ' (' + str(file_number+1) + '/' +str(self.number_of_files) + ')')
        self.raw_plot = ax1.plot(w[file_number], f[file_number], 'black',
                                 picker=5, zorder=1)
        ax1.set_xlim([bounds[0], bounds[1]])

        ax2.set_xlim([bounds[0], bounds[1]])
        ax2.plot([bounds[0], bounds[1]], [1, 1], 'turquoise', zorder=999)
        ax2.set_ylim([0, 1.2])

        self.lastind = 0
        self.vertical_x_cen = []

        self.wavelengths = w
        self.fluxes = f
        self.ax1 = ax1
        self.ax2 = ax2
        self.fig = fig

        self.file_number = file_number

        self.w_min = max([i[0] for i in w])
        self.w_max = min([i[-1] for i in w])
        self.lower_bound = bounds[0]
        self.upper_bound = bounds[1]
        self.spline_range = np.arange(self.w_min, self.w_max+1, 1)
        self.filepath = filepath
        self.knot_width = 5.0
        self.knot_half_width = self.knot_width/2
        self.flux_knots = []

    def onpress(self, event):
        '''
        n - normalize and save knots specific to spectrum
        d - delete knot point closest to cursor
        enter - fit and display spline
        u - update the user on how many points have been selected
        m - saves knot points
        r - reads in saved knot points
        e - reads in saved knots for specific spectrum if available
        q - saves knots for specific spectrum
        '''
        if self.lastind is None:
            return
        if event.key not in ('n', 'd', 'enter', 'u', 'm', 'r', 'e', 'q'):
            return
        if event.key == 'n':
            # normalizes and saves knots
            self.normalize()

            self.vertical_x_cen.sort()
            np.savetxt(self.files[self.file_number].split('.fits')
                       [0] + '_knots.txt', self.vertical_x_cen)

            self.file_number += 1

            if self.file_number < self.number_of_files:
                self.reset_plots()
            else:
                plt.close(self.fig)
                print("-----------------------------------")
                print("Done normalizing " + str(self.number_of_files) + " spectra.")


        elif event.key == 'enter':
            # print('fitting spline...')
            self.fit_spline()
            # print('spline fit successful')
        elif event.key == 'r':
            try:  # reads in knots if available
                print("Trying to read in saved knots.")
                x = np.loadtxt(self.files[self.file_number].split('.fits')[0]
                               + '_knots.txt')
                self.vertical_x_cen = list(x)
                self.vertical_x_cen.sort()
                self.flux_knots = self.determine_spline_pairs(
                    self.wavelengths[self.file_number],
                    self.fluxes[self.file_number])
            except Exception:
                print('Could not find knots file.')
                pass
        elif event.key == 'e':
            infname = self.files[self.file_number]
            try:
                print("Trying to read in already excisting knots...")
                starname = infname.split('_')[0] + '_' + infname.split('_')[1]
                starname = starname.split('raw/')[1]
                infname = glob.glob('./infiles_normed/' + starname + '*txt')[0]
                
                if os.path.isfile(infname) is False:
                    infname = (self.files[self.file_number].split('.fits')[0]
                                    + '_knots.txt')
                x = (np.loadtxt(infname))
                self.vertical_x_cen = list(x)
                self.vertical_x_cen.sort()
                self.flux_knots = self.determine_spline_pairs(
                    self.wavelengths[self.file_number],
                    self.fluxes[self.file_number])
            except Exception:
                print("Knots could not be loaded.")
                pass
        elif event.key == 'q':
            self.vertical_x_cen.sort()
            np.savetxt(self.files[self.file_number].split('.fits')[0]
                       + '_knots.txt', self.vertical_x_cen)
        elif event.key == 'm':
            self.vertical_x_cen.sort()
            np.savetxt(self.filepath + '/knots.txt', self.vertical_x_cen)
        elif event.key == 'u':  # updates on number of knots currently selected
            print(len(self.vertical_x_cen))
        elif event.key == 'd':
            # delets last knot
            try:
                xe = event.xdata
                dif = abs(self.vertical_x_cen - xe)
                dif_ind = list(dif).index(min(dif))
                del self.vertical_x_cen[dif_ind]
                del self.flux_knots[dif_ind]
                self.flux_knots = [f for _, f in
                                   sorted(zip(self.vertical_x_cen,
                                              self.flux_knots))]
                self.vertical_x_cen.sort()
            except Exception:
                pass

        self.update()

    def onpick(self, event):
#         if self.fig.canvas.manager.toolbar._active is None:
        if self.ax1.get_navigate_mode() == None:
            xe = event.xdata
            if self.w_min < xe < self.w_max:
                self.vertical_x_cen.append(xe)
            self.vertical_x_cen = [i for i in self.vertical_x_cen if
                                   i is not None]
            wavelength = self.wavelengths[self.file_number]
            flux = self.fluxes[self.file_number]
            inds = [i for i in range(len(wavelength)) if xe -
                    self.knot_half_width <=
                    wavelength[i] <= xe + self.knot_half_width]
            self.flux_knots.append(np.median(flux[inds]))
            self.flux_knots = [f for _, f in sorted(zip(self.vertical_x_cen,
                                                        self.flux_knots))]
            self.vertical_x_cen.sort()
            self.update()

    def update(self):
        if self.lastind is None:
            return
        try:
            self.knots.remove()
        except Exception:
            pass
        try:
            for i in self.knots:
                i.remove()
        except Exception:
            pass
        try:
            for i in self.knot_pairs:
                self.ax1.lines.remove(i)
        except Exception:
            pass

        self.knots = []

        for i in self.vertical_x_cen:
            self.knot = self.ax1.axvspan(i-self.knot_half_width,
                                         i + self.knot_half_width, alpha=0.3,
                                         color='C1')
            self.knots.append(self.knot)

        self.knot_pairs = self.ax1.plot(self.vertical_x_cen, self.flux_knots,
                                        'x', c='C1')

        self.fig.canvas.draw()

    '''------SPLINE FITTING FUNCTIONS-------'''

    def determine_spline_pairs(self, wavelength, flux):
        final_prespline_fluxes = []
        for cen in self.vertical_x_cen:
            inds = [i for i in range(len(wavelength)) if cen -
                    self.knot_half_width <=
                    wavelength[i] <= cen + self.knot_half_width]
            final_prespline_fluxes.append(np.median(flux[inds]))
        return final_prespline_fluxes

    def fit_spline(self):
        try:
            for i in self.spline_plot:
                self.ax1.lines.remove(i)
            for i in self.norm_plot:
                self.ax2.lines.remove(i)
        except Exception:
            pass
        # self.vertical_x_cen.sort()

        wave_knots = self.spline_range
        spline_solution = splrep(self.vertical_x_cen, self.flux_knots, k=2)
        self.spline_solution = spline_solution
        spline = splev(wave_knots, spline_solution)

        self.spline_plot = self.ax1.plot(wave_knots, spline, 'darkorange')

        spline_norm = splev(self.wavelengths[self.file_number],
                            spline_solution)
        norm_flux = self.fluxes[self.file_number]/spline_norm

        self.norm_plot = self.ax2.plot(self.wavelengths[self.file_number],
                                       norm_flux, 'black')

    '''------NORMALIZATION ROUTINE------------'''

    def normalization_fit_spline(self, wavelength, flux, ind_range):
        self.vertical_x_cen.sort()
        spline_solution = splrep(self.vertical_x_cen, self.flux_knots, k=2)

        spline = splev(wavelength[ind_range], spline_solution)
        self.ax1.plot(wavelength[ind_range], spline)
        return wavelength[ind_range], spline

    def normalize(self):
        print('Normalizing...')
        wave = self.wavelengths[self.file_number]
        flu = self.fluxes[self.file_number]
        spline_norm = splev(wave, self.spline_solution)
        norm_flux = flu/spline_norm

        cur_file = self.files[self.file_number]
        outfilename = outfolder + str(cur_file).split('/')[-1].split('.fit')[0] + '_norm.fits'

        if os.path.isfile(outfilename):
            print("Spectrum already normalized.")
        else:
            write_hermes(cur_file, norm_flux, outfilename)
            print('Saved:  ' + outfilename.split('/')[-1])

    def reset_plots(self):
        try:
            for i in self.spline_plot:
                self.ax1.lines.remove(i)
            for i in self.norm_plot:
                self.ax2.lines.remove(i)
            for i in self.raw_plot:
                self.ax1.lines.remove(i)
        except Exception:
            pass
        try:
            x = np.loadtxt(self.filepath + '/knots.txt')
            self.vertical_x_cen = list(x)
            self.vertical_x_cen.sort()
        except Exception:
            pass

 #       plot_title = self.files[self.file_number].split('/')[2].split('_202')[0].replace('_', ' ')
  #      self.ax1.set_title(plot_title + ' (' + str(self.file_number+1) + '/' +
  #                         str(self.number_of_files) + ')')
#        self.ax1.set_title('Spectrum %i of %i' % (self.file_number+1,
#                                                  self.number_of_files))
        self.raw_plot = self.ax1.plot(
            self.wavelengths[self.file_number], self.fluxes[self.file_number],
            'black',  picker=5, zorder=1)
        self.ax1.set_xlim([self.lower_bound, self.upper_bound])
        #self.ax1.set_ylim([0, 1.1])

        self.ax2.set_xlim([self.lower_bound, self.upper_bound])
        self.ax2.set_ylim([0, 1.2])

        self.flux_knots = self.determine_spline_pairs(
            self.wavelengths[self.file_number], self.fluxes[self.file_number])
        self.fit_spline()


def picker(spec_directory=None, spec_file=None):
    if spec_directory:
        files = glob.glob(spec_directory + '/*.fits')
    elif spec_file:
        files = [spec_file]
    files.sort()
    file_number = 0
    number_of_files = len(files)

    fig = plt.figure(1)
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)

    wavelengths = []
    fluxes = []
    for i in range(number_of_files):
        outfilename = outfolder + str(files[i]).split('.fits')[0] + '_norm.fits'

        if os.path.isfile(outfilename):
            print("Spectrum already normalized.")
        else:
            w, f = read_HERMES(files[i])
            if len(f) < 10:
                f = f[0]
            wavelengths.append(w)
            fluxes.append(f)

    bounds = [min(w), max(w)]
    browser = PointBrowser(fig, ax1, ax2, wavelengths, fluxes,
                           spec_directory, files, file_number, bounds)

    fig.canvas.mpl_connect('button_press_event', browser.onpick)
    fig.canvas.mpl_connect('key_press_event', browser.onpress)

    plt.show()


def scale_raw_spectra(w, f, window=[7500, 7550]):
    inds = [i for i in range(len(w)) if window[0] <= w[i] <= window[1]]
    return f / np.mean(f[inds])


def print_help():
    print("")
    print("****************************************")
    print("* HERMES Binary Spectrum Normalization *")
    print("****************************************")
    print("")
    print('n - normalize and save knots specific to spectrum')
    print('d - delete knot point closest to cursor')
    print('enter - fit and display spline')
    print('u - update the user on how many points have been selected')
    print('m - saves knot points')
    print('r - reads in saved knot points')
    print('e - reads in saved knots for specific spectrum if available')
    print('q - saves knots for specific spectrum')


infolder = './'  # folder with the spectra to normalize
outfolder = 'norm/'  # folder to save output spectra in, should exist beforehand

print_help()
picker(infolder)
# picker(spec_file='./test.fits')
# picker(spec_directory = '')
