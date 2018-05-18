package ampcontrol.audio.asio;

import ampcontrol.audio.AudioInputBuffer;
import ampcontrol.audio.ClassifierInputProvider;
import ampcontrol.audio.ClassifierInputProviderFactory;
import ampcontrol.audio.Cnn2DInputProvider;
import ampcontrol.audio.processing.ProcessingFactoryFromString;
import ampcontrol.audio.processing.ProcessingResult;
import com.beust.jcommander.Parameter;
import com.synthbot.jasiohost.AsioChannel;
import com.synthbot.jasiohost.AsioDriver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * {@link ClassifierInputProviderFactory} which creates input from a given channel on a given ASIO device.
 * 
 * @author Christian Skärby
 */
public class AsioClassifierInputFactory implements ClassifierInputProviderFactory {

    private static final Logger log = LoggerFactory.getLogger(AsioClassifierInputFactory.class);


    @Parameter(names = "-asioDriver", description = "Asio driver to use")
    private String driverName = "";

    @Parameter(names = "-channel", description = "Input channel to classify")
    private int channelInd = 0;

    @Parameter(names = "-samplingRate", description = "Sample rate used when training model")
    private int sampleRate = 44100;

    // Should be final but can't thanks to JCommander...
    private AsioDriver driver;

    private final Map<AudioInputBuffer, Map<String, ClassifierInputProvider.Updatable>> inputProviderCache = new LinkedHashMap<>();
    private final Map<Integer, AudioInputBuffer> audioInputCache = new LinkedHashMap<>();
    private final Map<Integer, AsioChannel> activeAsioChannels = new HashMap<>();

    public void initialize() {
        // Effectively untestable as Asio stuff is not mockable
        AsioDriver.getDriverNames().forEach(name -> log.info("Found ASIO driver with name: " + name));
        driver = AsioDriver.getDriver(driverName);

        // No matter what, we must shut down the driver when program ends!
        setShutdownHook(driver);
        driver.setSampleRate(sampleRate);
    }

    private void setShutdownHook(AsioDriver driver) {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            log.info("Program ended, tear down driver");
            try{
                driver.stop();
            } finally {
                driver.shutdownAndUnloadDriver();
            }
        }));
    }

    @Override
    public ClassifierInputProvider createInputProvider(String inputDescriptionString) {
        final AudioInputBuffer audioInput = createAudioBuffer(inputDescriptionString);

        final Map<String, ClassifierInputProvider.Updatable> processingToInputProvider = inputProviderCache
                .computeIfAbsent(audioInput, k -> new HashMap<>());

        final ProcessingResult.Factory resultFactory = new ProcessingFactoryFromString(driver.getSampleRate())
                .get(inputDescriptionString);

        return processingToInputProvider
                .computeIfAbsent(resultFactory.name(), k -> new Cnn2DInputProvider(audioInput, () -> resultFactory));

    }

    @Override
    public ClassifierInputProvider.UpdateHandle finalizeAndReturnUpdateHandle() {
        driver.createBuffers(new HashSet<>(activeAsioChannels.values()));
        driver.start();
        return () -> inputProviderCache.values().stream()
                .flatMap(procToInputMap -> procToInputMap.values().stream())
                .forEach(ClassifierInputProvider.UpdateHandle::updateInput);
    }

    private AudioInputBuffer createAudioBuffer(String str) {
        final int windowSizeMs = ClassifierInputProviderFactory.parseWindowSize(str);
        final int nrofSamples = (int)(driver.getSampleRate() * windowSizeMs / 1000);
        final int key = hashKey(nrofSamples,channelInd);
        return audioInputCache.computeIfAbsent(key, k -> {

            final AsioChannel channel = activeAsioChannels.computeIfAbsent(channelInd,  k2 -> driver.getChannelInput(channelInd));

            AsioAudioInputBuffer asioInputBuffer = new AsioAudioInputBuffer(
                    new SingleAsioInputChannel(channel),
                    driver.getBufferPreferredSize(),
                    nrofSamples);
            driver.addAsioDriverListener(asioInputBuffer);
            return  asioInputBuffer;

        });
    }

    private static int hashKey(int... ints) {
        return Arrays.hashCode(ints);
    }
}
