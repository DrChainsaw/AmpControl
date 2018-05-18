package ampcontrol.admin.service.control.mqtt;

import ampcontrol.admin.service.control.AppControlService;
import ampcontrol.admin.service.control.ControlRegistry;
import ampcontrol.admin.service.control.MessageSubscriptionRegistry;
import ampcontrol.admin.service.control.TopicPublisher;
import com.beust.jcommander.Parameter;
import org.eclipse.paho.client.mqttv3.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Service for interacting with the application through MQTT.
 *
 * @author Christian Skärby
 */
public class MqttAppControlService implements TopicPublisher, AppControlService {

    private static final Logger log = LoggerFactory.getLogger(MqttAppControlService.class);

    @Parameter(names = "-mqttServer", description = "MQTT server to connect to")
    private String server = "";

    @Parameter(names = "-mqttUsername", description = "MQTT server username. Leave empty if none")
    private String uname = "";

    @Parameter(names = "-mqttPwd", description = "MQTT server password. Leave empty if none")
    private String pwd = "";

    @Parameter(names = "-mqttListenTopic", description = "MQTT topic to subscribe to")
    private String listenTopic = "podxtcontrol/commands";

    @Parameter(names = "-mqttOnlineTopic", description = "MQTT topic to publish to when service goes online/offline")
    private String onlinePublishTopic = "podxtcontrol/online";

    private final static MqttMessage ON = new MqttMessage("ON".getBytes());
    private final static MqttMessage OFF = new MqttMessage("OFF".getBytes());

    private ClientFactory clientFactory = serverUri -> new MqttClient(serverUri, MqttClient.generateClientId());
    private IMqttClient client;

    /**
     * Interface for client creation. Intended for testing
     */
    interface ClientFactory {
        /**
         * Create a new {@link IMqttClient} to the given serverUri
         *
         * @param serverUri
         * @return
         * @throws MqttException
         */
        IMqttClient create(String serverUri) throws MqttException;
    }

    /**
     * Starts the service by connecting to the broker. Returns a {@link MessageSubscriptionRegistry} for users to define
     * actions in response to messages on the listenTopic.
     *
     * @return Returns a {@link MessageSubscriptionRegistry}
     * @throws MqttException
     */
    @Override
    public ControlRegistry start() throws MqttException {

        client = clientFactory.create(server); //new MqttClient(server, MqttClient.generateClientId());
        MqttCallbackMap callbackMapper = new MqttCallbackMap(topic -> {
            try {
                client.subscribe(topic);
            } catch (MqttException e) {
                log.warn(e.toString());
            }
        });
        MqttConnectOptions connectOptions = new MqttConnectOptions();
        connectOptions.setCleanSession(true);

        connectOptions.setUserName(uname);
        connectOptions.setPassword(pwd.toCharArray());
        connectOptions.setWill(onlinePublishTopic, OFF.getPayload(), 0, false);

        client.connect(connectOptions);
        client.publish(onlinePublishTopic, ON);

        client.setCallback(callbackMapper);
        client.subscribe(listenTopic);
        return callbackMapper;
    }

    /**
     * Disconnects from the broker.
     *
     * @throws MqttException
     */
    @Override
    public void stop() throws MqttException {
    	client.publish(onlinePublishTopic, OFF);
    	client.disconnect();
    }

    @Override
    public void publish(String topic, String message) {
        try {
        client.publish(topic, new MqttMessage(message.getBytes()));
        } catch (MqttException e) {
            System.err.println("Classification message failed!");
            e.printStackTrace();
        }
    }

    /**
     * Sets the {@link ClientFactory} to use. Intended for testing
     *
     * @param clientFactory factory to use
     */
    void setClientFactory(ClientFactory clientFactory) {
        this.clientFactory = clientFactory;
    }
}
