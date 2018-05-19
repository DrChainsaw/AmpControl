package ampcontrol.audio.asio;

import ampcontrol.audio.AudioInputBuffer;
import com.synthbot.jasiohost.AsioChannel;
import com.synthbot.jasiohost.AsioDriver;
import com.synthbot.jasiohost.AsioDriverListener;

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * A circular buffer holding the last desiredNrofSamples from an {@link AsioChannel}.
 * Use for online classification of audio from an ASIO device.
 *
 * @author Christian Skärby
 */
public class AsioAudioInputBuffer implements AsioDriverListener, AudioInputBuffer {

    private final AsioInputChannel inputChannel;
    private float[] bufferRetreiver;
    private final double[] samples;
    private final int desiredNrofSamples;
    private int cursor = 0;

    AsioAudioInputBuffer(AsioInputChannel inputChannel, int bufferSize, int desiredSamples) {
        this.inputChannel = inputChannel;
        this.bufferRetreiver = new float[bufferSize];
        this.samples = new double[desiredSamples];
        desiredNrofSamples = desiredSamples;
        if (desiredSamples <= 0) {
            throw new IllegalArgumentException("Desired number of samples must be > 0!!");
        }
    }

    @Override
    public double[] getAudio() {
        double[] audio = new double[desiredNrofSamples];
        synchronized (this) {
            for (int i = 0; i < audio.length; i++) {
                int samplesInd = getSamplesInd(i);
                audio[i] = samples[samplesInd];
            }
        }
        return audio;
    }


    @Override
    public void sampleRateDidChange(double v) {
        //Ignore
    }

    @Override
    public void resetRequest() {
        //Ignore
    }

    @Override
    public void resyncRequest() {
        //Ignore
    }

    @Override
    public void bufferSizeChanged(int i) {
        bufferRetreiver = new float[i];
    }

    @Override
    public void latenciesChanged(int i, int i1) {
        //Ignore
    }

    @Override
    public void bufferSwitch(long l, long l1, Set<AsioChannel> set) {
        if (inputChannel.updateBuffer(bufferRetreiver, set)) {
            synchronized (this) {
                for (int i = 0; i < bufferRetreiver.length; i++) {
                    int samplesInd = getSamplesInd(i);
                    // Models trained in this framework use shorts for waveinput.
                    // More "accurate" to cast to double before multiplying with a large value?
                    samples[samplesInd] = ((double) bufferRetreiver[i] * Short.MAX_VALUE);
                }
                cursor += bufferRetreiver.length;
                cursor %= samples.length;
            }
        }
    }

    private int getSamplesInd(int i) {
        return (i + cursor) % samples.length;
    }

    public static void main(String[] args) {
        List<String> driverNameList = AsioDriver.getDriverNames();
        System.out.println(driverNameList);
        AsioDriver driver = null;
        try {
            driver = AsioDriver.getDriver(driverNameList.get(0));

            AsioChannel channel = driver.getChannelInput(0);
            System.out.println("size; " + driver.getBufferPreferredSize());
            System.out.println("max; " + driver.getBufferMaxSize());
            System.out.println("gran: " + driver.getBufferGranularity());

            Set<AsioChannel> activeChannels = Collections.singleton(channel);

            AsioAudioInputBuffer stream = new AsioAudioInputBuffer(new SingleAsioInputChannel(channel), driver.getBufferPreferredSize(), 4410);
            driver.addAsioDriverListener(stream);
            driver.createBuffers(activeChannels);

            driver.start();

            Thread.sleep(2000);


            // System.out.println("short: " + Arrays.toString(stream.getAudio2(4410)));
            // System.out.println("short: " + Arrays.toString(stream.getAudio(4410)));


        } catch (Exception ie) {
            ie.printStackTrace(System.err);
        } finally {
            Objects.requireNonNull(driver).shutdownAndUnloadDriver();
        }
    }

}
