package ampcontrol.audio.processing;

/*
  Please feel free to use/modify this class.
  If you give me credit by keeping this information or
  by sending me an email before using it or by reporting bugs , i will be happy.
  Email : gtiwari333@gmail.com,
  Blog : http://ganeshtiwaridotcomdotnp.blogspot.com/
 */

import java.util.stream.Stream;

/**
 * Mel-Frequency Spectrum Coefficients.
 *
 * @author Ganesh Tiwari
 * @author Hanns Holger Rutz
 */
public class Mfsc implements ProcessingResult.Factory {

    private final static double lowerFilterFreq = 80.00; // FmelLow

    private final double sampleRate;
    private final double upperFilterFreq;


    public Mfsc(double sampleRate) {
        this.sampleRate = sampleRate;
        upperFilterFreq = sampleRate / 2.0;
    }

    @Override
    public String name() {
        return nameStatic();
    }

    public static String nameStatic() {
        return "mfsc";
    }

    @Override
    public ProcessingResult create(ProcessingResult input) {
        return new Result(input);
    }

    private final class Result implements ProcessingResult {

        private final ProcessingResult input;

        public Result(ProcessingResult input) {
            this.input = input;
        }

        @Override
        public Stream<double[][]> stream() {
            return input.stream().map(inputArr -> {
                final int nrofFrames = inputArr.length;
                final int nrofSamplesPerFrame = inputArr[0].length;
                final double[][] mfcc = new double[nrofFrames][nrofSamplesPerFrame];
                final int numMelFilters = nrofSamplesPerFrame;
                final int samplesPerFrame = nrofSamplesPerFrame;
                for (int i = 0; i < nrofFrames; i++) {
                    mfcc[i] = process(inputArr[i], numMelFilters, samplesPerFrame);
                }
                return mfcc;
            });
        }
    }

    private double[] process(double[] bin, int numMelFilters, int samplesPerFrame) {
        /*
         * cBin=frequencies of the channels in terms of FFT bin indices (cBin[i]
         * for the i -th channel)
         */

        // prepare filter for for melFilter
        final int cBin[] = fftBinIndices(numMelFilters, samplesPerFrame);// same for all
        // process Mel filter bank
        final double fBank[] = melFilter(bin, cBin, numMelFilters);
        // magnitudeSpectrum and bin filter indices

        // System.out.println("after mel filter");
        // ArrayWriter.printDoubleArrayToConsole(fBank);

        // Non-linear transformation
        final double f[] = nonLinearTransformation(fBank);
        // System.out.println("after N L T");
        // ArrayWriter.printDoubleArrayToConsole(f);

        // Cepstral coefficients, by DCT
        // System.out.println("after DCT");
        // ArrayWriter.printDoubleArrayToConsole(cepc);
        //return perform(f);
        return f;
    }

    private int[] fftBinIndices(int numMelFilters, int samplesPerFrame) {
        final int cBin[] = new int[numMelFilters + 2];
        cBin[0] = (int) Math.round(lowerFilterFreq / sampleRate * samplesPerFrame);// cBin0
        cBin[cBin.length - 1] = (samplesPerFrame / 2);// cBin24
        for (int i = 1; i <= numMelFilters; i++) {// from cBin1 to cBin23
            final double fc = centerFreq(i, numMelFilters);// center freq for i th filter
            cBin[i] = (int) Math.round(fc / sampleRate * samplesPerFrame);
        }
        return cBin;
    }

    /**
     * Performs mel filter operation
     *
     * @param bin  magnitude spectrum (| |) of fft
     * @param cBin mel filter coefficients
     * @return mel filtered coefficients --> filter bank coefficients.
     */
    private double[] melFilter(double bin[], int cBin[], int numMelFilters) {
        final double temp[] = new double[numMelFilters + 2];
        for (int k = 1; k <= numMelFilters; k++) {
            double num1 = 0.0;
            double num2 = 0.0;
            for (int i = cBin[k - 1]; i <= cBin[k]; i++) {
                num1 += ((i - cBin[k - 1] + 1) / (cBin[k] - cBin[k - 1] + 1)) * bin[i];
            }

            for (int i = cBin[k] + 1; i <= cBin[k + 1]; i++) {
                num2 += (1 - ((i - cBin[k]) / (cBin[k + 1] - cBin[k] + 1))) * bin[i];
            }

            temp[k] = num1 + num2;
        }
        final double fBank[] = new double[numMelFilters];
        System.arraycopy(temp, 1, fBank, 0, numMelFilters);
        return fBank;
    }

    /**
     * performs nonlinear transformation
     *
     * @param fBank filter bank coefficients
     * @return f log of filter bac
     */
    private double[] nonLinearTransformation(double fBank[]) {
        double f[] = new double[fBank.length];
        final double FLOOR = -50;
        for (int i = 0; i < fBank.length; i++) {
            f[i] = Math.log(fBank[i]);
            // check if ln() returns a value less than the floor
            if (f[i] < FLOOR) {
                f[i] = FLOOR;
            }
        }
        return f;
    }

    private double centerFreq(int i, int numMelFilters) {
        final double melFLow = freqToMel(lowerFilterFreq);
        final double melFHigh = freqToMel(upperFilterFreq);
        final double temp = melFLow + ((melFHigh - melFLow) / (numMelFilters + 1)) * i;
        return inverseMel(temp);
    }

    private double inverseMel(double x) {
        final double temp = Math.pow(10, x / 2595) - 1;
        return 700 * (temp);
    }

    private double freqToMel(double freq) {
        return 2595 * log10(1 + freq / 700);
    }

    private double log10(double value) {
        return Math.log(value) / Math.log(10);
    }


}