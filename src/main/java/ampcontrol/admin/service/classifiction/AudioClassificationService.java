package ampcontrol.admin.service.classifiction;

import ampcontrol.admin.service.Service;
import ampcontrol.admin.service.control.SubscriptionRegistry;
import ampcontrol.amp.ClassificationListener;
import ampcontrol.audio.ClassifierInputProvider;
import ampcontrol.model.inference.Classifier;
import com.beust.jcommander.Parameter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.time.Duration;
import java.time.LocalTime;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Service for doing online audio classification. Periodically asks all connected {@link Classifier Classifiers}
 * to classify their input and forwards the classification to connected
 * {@link ClassificationListener ClassificationListeners}
 *
 * @author Christian Skärby
 */
public class AudioClassificationService implements Service {

    private static final Logger log = LoggerFactory.getLogger(AudioClassificationService.class);
    
    @Parameter(names = {"-minTimeBetweenUpdates", "-mtbu"}, description = "Shortest time between classifications")
	private int minTimeBetweenUpdatesMs = 100;

    @Parameter(names = "-actAutoMsg", description = "Message contents to start auto program change")
    private String actMsg = "activateAutoProgramChange";

    @Parameter(names = "-dectAutoMsg", description = "Message contents to stop auto program change")
    private String deactMsg = "deactivateAutoProgramChange";
	
    private boolean isInit = false;

    private ClassificationListener ampInterface;
    private Classifier classifier;
    private ClassifierInputProvider.UpdateHandle inputUpdateHandle;
    private ScheduledExecutorService executorService;
    
    private LocalTime lastTime = LocalTime.now();

    /**
     * Initialize the service. Reason for this method instead of a constructor is only because Jcommander must have
     * an instance to set parameters.
     *
     * @param ampInterface
     * @param classifier
     * @param inputUpdateHandle
     * @throws IOException
     */
    public void initialize(
    		ClassificationListener ampInterface,
    		Classifier classifier,
    		ClassifierInputProvider.UpdateHandle inputUpdateHandle) {

        this.ampInterface = ampInterface;
        this.classifier = classifier;
        this.inputUpdateHandle = inputUpdateHandle;
        inputUpdateHandle.updateInput();
        log.info("" + classifier.classify());
        isInit = true;
    }

    /**
     * Start the service. Classifiction will be scheduled every minTimeBetweenUpdatesMs and result will be provided to
     * listeners.
     */
    private void start() {
        if (!isInit) {
            throw new IllegalStateException("Must be initialized before starting!");
        }
        log.info("Start switch service. Is running: " + isRunning());
        if(isRunning()) {
            return;
        }

        executorService = Executors.newSingleThreadScheduledExecutor();
        // Main loop
        executorService.scheduleAtFixedRate(() -> {     	
        	 // Step 1, update input
            inputUpdateHandle.updateInput();

            // Step 2: classify
            final LocalTime starttime = LocalTime.now();
            INDArray classification = classifier.classify();
            final LocalTime endtime = LocalTime.now();
            log.info("classTime: " + Duration.between(starttime,endtime).toMillis());
            log.info(Duration.between(lastTime, endtime).toMillis() + " classification: " + classification);
            lastTime = endtime;
            
            //Step 3: Give classification to listener
            ampInterface.indicateAudioClassification(classification);
        }, 0, TimeUnit.MILLISECONDS.toNanos(minTimeBetweenUpdatesMs), TimeUnit.NANOSECONDS);
           
    }

    /**
     * Stops the service. No classification will be done.
     */
    @Override
    public void stop() {
    	if(isRunning()) {
    		executorService.shutdown();
    		log.info("Thread stopped " + executorService.isShutdown());
    		executorService = null;
    	}
    }

    /**
     * True if service is running
     *
     * @return true if service is running
     */
    public boolean isRunning() {
    	return executorService != null && !executorService.isShutdown();
    }

    @Override
    public void registerTo(SubscriptionRegistry subscriptionRegistry) {
        subscriptionRegistry.registerSubscription(actMsg, () -> start());
        subscriptionRegistry.registerSubscription(deactMsg, () -> stop());
    }
}

