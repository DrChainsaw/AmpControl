package ampcontrol.admin.service.control.mqtt;

import com.beust.jcommander.JCommander;
import org.eclipse.paho.client.mqttv3.MqttException;
import org.junit.Test;

import java.util.Collections;

import static org.junit.Assert.*;

/**
 * Test cases for {@link MqttAppControlService}
 *
 * @author Christian Skärby
 */
public class MqttAppControlServiceTest {

    /**
     * Test that service can be stared and stopped
     */
    @Test
    public void startStop() {
        final MockMqttClient client = new MockMqttClient();
        final MqttAppControlService service = new MqttAppControlService();
        service.setClientFactory(server -> client);

        final String onlineTopic = "olTopic";
        final String commandTopic = "cmdTopic";
        final String argString = "-mqttServer dummyServer -mqttUsername uname -mqttPwd pwd -mqttOnlineTopic "
                + onlineTopic + " -mqttListenTopic " + commandTopic;
        JCommander.newBuilder().addObject(service)
                .build()
                .parse(argString.split(" "));

        try {
            service.start();
            assertTrue("Shall be connected!", client.isConnected());
            assertEquals("Incorrect subscribed topics!", Collections.singleton(commandTopic), client.getSubscribedTopics());
            assertEquals("Incorrect published topic!", onlineTopic, client.getLastPublishedTopic());
            assertEquals("Incorrect published message!", "ON", client.getLastPublishedMessage());
            client.clearLastPublished();

            service.stop();
            assertFalse("Shall be connected!", client.isConnected());
            assertEquals("Incorrect published topic!", onlineTopic, client.getLastPublishedTopic());
            assertEquals("Incorrect published message!", "OFF", client.getLastPublishedMessage());
            client.clearLastPublished();

            service.start();
            assertTrue("Shall be connected!", client.isConnected());
            assertEquals("Incorrect subscribed topics!", Collections.singleton(commandTopic), client.getSubscribedTopics());
            assertEquals("Incorrect published topic!", onlineTopic, client.getLastPublishedTopic());
            assertEquals("Incorrect published message!", "ON", client.getLastPublishedMessage());
            client.clearLastPublished();

        } catch (MqttException e) {
            fail("Exception thrown! Message: \n" + e.getMessage());
        }
    }
}