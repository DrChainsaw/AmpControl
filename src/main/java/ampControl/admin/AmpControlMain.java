package ampControl.admin;

import java.util.Map;

import com.beust.jcommander.JCommander;

import ampControl.admin.service.classifiction.AudioClassificationService;
import ampControl.amp.PublishingClassificationListener;
import ampControl.admin.service.control.mqtt.MqttAppControlService;
import ampControl.amp.ClassificationListener;
import ampControl.audio.asio.AsioClassifierInputFactory;
import ampControl.model.inference.Classifier;
import ampControl.model.inference.ClassifierFromParameters;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;

/**
 * Main class for doing AmpControl. Just does initialization. Actual main loop is handled by the {@link Engine}.
 *
 * @author Christian Skärby
 */
public class AmpControlMain {

    public  static void main(String[] args) {

        // Class not under test. Is it even testable in practice?
        final Engine engine = new Engine();
        final MqttAppControlService mqttAppControlService = new MqttAppControlService();
        final PublishingClassificationListener.Factory mqttClassificationListenerFactory =
                new PublishingClassificationListener.Factory(mqttAppControlService);
        final AudioClassificationService audioClassificationService = new AudioClassificationService();
        final ClassifierFromParameters classifierFromParameters = new ClassifierFromParameters();
        final AsioClassifierInputFactory inputProviderFactory = new AsioClassifierInputFactory();

        JCommander.Builder jcBuilder = JCommander.newBuilder()
                .addObject(new Object[] {
                        engine,
                        mqttAppControlService,
                        audioClassificationService,
                        inputProviderFactory,
                        classifierFromParameters,
                        mqttClassificationListenerFactory});

        Map<String, ClassificationListener.Factory> ampFactoryCommands = ClassificationListener.getFactoryCommands();
        ampFactoryCommands.entrySet().forEach(entry -> jcBuilder.addCommand(entry.getKey(), entry.getValue()));

        JCommander jc = jcBuilder.build();
        jc.parse(args);

        final ClassificationListener ampInterface = ampFactoryCommands.get(jc.getParsedCommand()).create();
        final ClassificationListener mqttInterface = mqttClassificationListenerFactory.create();
        final ClassificationListener classificationListenerAgg = arr -> {
        	ampInterface.indicateAudioClassification(arr);
            mqttInterface.indicateAudioClassification(arr);
        };
        inputProviderFactory.initialize();

        try {
            // Might need to move into concrete Classifiers if something else is used in training
            DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);
            final Classifier classifier = classifierFromParameters.getClassifier(inputProviderFactory);
            audioClassificationService.initialize(
                    classificationListenerAgg,
                    classifier,
                    inputProviderFactory.finalizeAndReturnUpdateHandle());
            engine.initialize(mqttAppControlService, audioClassificationService);
            engine.run();
        } catch (Exception e) {
            System.out.println("Application failed!");
            e.printStackTrace();
            System.exit(1);
        }

    }
}
