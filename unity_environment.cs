using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

/*
Behaviour Parameters:
Vector Obs -> Space Size = 9
Discrete Braches = 1; Branch Size = 4
*/

public class agentAgent : Agent
{
    const int k_NoAction = 0;
    const int k_Up = 1;
    const int k_Down = 2;
    const int k_Right = 3;
    
    public float speed;
    public Transform End_Goal;
    public float prev_goal_distance;


    // Obstacle 1 variables
    public Transform Obstacle_1;
    public float prev_up_distance_1;
    public float prev_down_distance_1;
    public Vector3 target_up_1;
    public Vector3 target_down_1;
    public int obstacle_direction_1;
    public Vector3 obstacle_destination_1;
    public Vector3 obstacle_pos_up_1;
    public Vector3 obstacle_pos_down_1;


    // Obstacle 2 variables
    public Transform Obstacle_2;
    public float prev_up_distance_2;
    public float prev_down_distance_2;
    public Vector3 target_up_2;
    public Vector3 target_down_2;
    public int obstacle_direction_2;
    public Vector3 obstacle_destination_2;
    public Vector3 obstacle_pos_up_2;
    public Vector3 obstacle_pos_down_2;


    // Obstacle 3 variables
    public Transform Obstacle_3;
    public float prev_up_distance_3;
    public float prev_down_distance_3;
    public Vector3 target_up_3;
    public Vector3 target_down_3;
    public int obstacle_direction_3;
    public Vector3 obstacle_destination_3;
    public Vector3 obstacle_pos_up_3;
    public Vector3 obstacle_pos_down_3;

    
    public override void OnEpisodeBegin()
    /* 
    This method is run when the environment is reset. In this method,
    the initial positions for the agent and end_goal are set and the 
    intial position and direction for the 3 obstacles are set.
    */

    {
        // speed of the obstacles
        speed = 1f;

        // select a random intitial position for agent
        int random_pos_agent = new System.Random().Next(3, 13);
        this.transform.localPosition = new Vector3(2f, 0.5f, random_pos_agent);

        // select a random position for end_goal
        int random_pos_goal = new System.Random().Next(3, 13);
        End_Goal.localPosition = new Vector3(33.5f, 0.5f, random_pos_goal);

        // calculate distance of agent from end_goal
        prev_goal_distance = Vector3.Distance(this.transform.localPosition, End_Goal.localPosition);


    // Obstacle 1

        // setting target locations on either side of obstacle_1 for the agent
        target_up_1 = new Vector3(11f, 0.5f, 14f);
        target_down_1 = new Vector3(11f, 0.5f, 1f);

        // calculating the distance between the agent and these targets
        prev_up_distance_1 = Vector3.Distance(this.transform.localPosition, target_up_1);
        prev_down_distance_1 = Vector3.Distance(this.transform.localPosition, target_down_1);

        // obstacle_1 oscillates between these two points
        obstacle_pos_up_1 = new Vector3(9f, 0.75f, 10f);
        obstacle_pos_down_1 = new Vector3(9f, 0.75f, 5f);

        // select a random obstacle direction (up or down)
        int random_dir_1 = new System.Random().Next(0, 2);
        
        // select direction up
        if (random_dir_1 > 0)
        {
            obstacle_direction_1 = 1;
            obstacle_destination_1 = obstacle_pos_up_1;
        }
        // select direction down
        else 
        {
            obstacle_direction_1 = -1;
            obstacle_destination_1 = obstacle_pos_down_1;
        }

        // select a random initial obstacle_1 position
        int random_pos_1 = new System.Random().Next(6, 10);
        Obstacle_1.localPosition = new Vector3(9f, 0.75f, random_pos_1);


    // Obstacle 2: Code is similar to the code for Obstacle 1

        target_up_2 = new Vector3(19.5f, 0.5f, 14f);
        target_down_2 = new Vector3(19.5f, 0.5f, 1f);
        prev_up_distance_2 = Vector3.Distance(this.transform.localPosition, target_up_2);
        prev_down_distance_2 = Vector3.Distance(this.transform.localPosition, target_down_2);
        obstacle_pos_up_2 = new Vector3(17.5f, 0.75f, 10f);
        obstacle_pos_down_2 = new Vector3(17.5f, 0.75f, 5f);

        int random_dir_2 = new System.Random().Next(0, 2);
        if (random_dir_2 > 0)
        {
            obstacle_direction_2 = 1;
            obstacle_destination_2 = obstacle_pos_up_2;
        }
        else 
        {
            obstacle_direction_2 = -1;
            obstacle_destination_2 = obstacle_pos_down_2;
        }

        int random_pos_2 = new System.Random().Next(6, 10);
        Obstacle_2.localPosition = new Vector3(17.5f, 0.75f, random_pos_2);


    // Obstacle 3: Code is similar to the code for Obstacle 1

        target_up_3 = new Vector3(28f, 0.5f, 14f);
        target_down_3 = new Vector3(28f, 0.5f, 1f);
        prev_up_distance_3 = Vector3.Distance(this.transform.localPosition, target_up_3);
        prev_down_distance_3 = Vector3.Distance(this.transform.localPosition, target_down_3);
        obstacle_pos_up_3 = new Vector3(26f, 0.75f, 10f);
        obstacle_pos_down_3 = new Vector3(26f, 0.75f, 5f);

        int random_dir_3 = new System.Random().Next(0, 2);
        if (random_dir_3 > 0)
        {
            obstacle_direction_3 = 1;
            obstacle_destination_3 = obstacle_pos_up_3;
        }
        else 
        {
            obstacle_direction_3 = -1;
            obstacle_destination_3 = obstacle_pos_down_3;
        }

        int random_pos_3 = new System.Random().Next(6, 10);
        Obstacle_3.localPosition = new Vector3(26f, 0.75f, random_pos_3);
    }



    void Update()
    /*
    This method is called during each step of the agent. This method controls
    the oscillation of the 3 obstacles.
    */
    {
    // Obstacle 1

        // obstacle reaches bottom -> move up
        if (Vector3.Distance(Obstacle_1.position, obstacle_pos_down_1) < 0.1f)
        {
            obstacle_direction_1 = 1;
            obstacle_destination_1 = obstacle_pos_up_1;
        }
        // obstacle reaches top -> move down
        else if (Vector3.Distance(Obstacle_1.position, obstacle_pos_up_1) < 0.1f)
        {
            obstacle_direction_1 = -1;
            obstacle_destination_1 = obstacle_pos_down_1;
        }
        // move obstacle_1
        Obstacle_1.position = Vector3.MoveTowards(Obstacle_1.position, obstacle_destination_1, speed * Time.deltaTime);


    // Obstacle 2

        // obstacle reaches bottom -> move up
        if (Vector3.Distance(Obstacle_2.position, obstacle_pos_down_2) < 0.1f)
        {
            obstacle_direction_2 = 1;
            obstacle_destination_2 = obstacle_pos_up_2;
        }
        // obstacle reaches top -> move down
        else if (Vector3.Distance(Obstacle_2.position, obstacle_pos_up_2) < 0.1f)
        {
            obstacle_direction_2 = -1;
            obstacle_destination_2 = obstacle_pos_down_2;
        }
        // move obstacle_2
        Obstacle_2.position = Vector3.MoveTowards(Obstacle_2.position, obstacle_destination_2, speed * Time.deltaTime);


    // Obstacle 3

        // obstacle reaches bottom -> move up
        if (Vector3.Distance(Obstacle_3.position, obstacle_pos_down_3) < 0.1f)
        {
            obstacle_direction_3 = 1;
            obstacle_destination_3 = obstacle_pos_up_3;
        }
        // obstacle reaches top -> move down
        else if (Vector3.Distance(Obstacle_3.position, obstacle_pos_up_3) < 0.1f)
        {
            obstacle_direction_3 = -1;
            obstacle_destination_3 = obstacle_pos_down_3;
        }
        // move obstacle_3
        Obstacle_3.position = Vector3.MoveTowards(Obstacle_3.position, obstacle_destination_3, speed * Time.deltaTime);
    }



    public override void CollectObservations(VectorSensor sensor)
    /*
    This method is called after each step. This method return 9 state features of
    the environment.
    */
    {
        // agent position
        sensor.AddObservation(this.transform.localPosition.x/15);
        sensor.AddObservation(this.transform.localPosition.z/15);

        // obstacles' direction and position
        sensor.AddObservation(obstacle_direction_1);
        sensor.AddObservation(Obstacle_1.localPosition.z/15);

        sensor.AddObservation(obstacle_direction_2);
        sensor.AddObservation(Obstacle_2.localPosition.z/15);

        sensor.AddObservation(obstacle_direction_3);
        sensor.AddObservation(Obstacle_3.localPosition.z/15);

        // end_goal position
        sensor.AddObservation(End_Goal.localPosition.z/15);
    }



    public override void OnActionReceived(ActionBuffers actionBuffers)
    /*
    This method is called when the environment receives an action. This method
    moves the agent according to the actions given and returns the reward for 
    each step.
    */
    {
        var action = actionBuffers.DiscreteActions[0];
        var next_position = transform.position;

        // distance to move agent
        const float move = 0.2f;

        // choose the new position of the agent after the action is taken
        switch (action)
        {
            case k_NoAction:
                break;
            case k_Up:
                next_position = transform.position + new Vector3(0, 0, move);
                break;
            case k_Down:
                next_position = transform.position + new Vector3(0, 0, -move);
                break;
            case k_Right:
                next_position = transform.position + new Vector3(move, 0, 0);
                break;
        }

        // prevent the agent from escaping the playing area
        if (next_position.z < 0.75f)
        {
            next_position.z = 0.75f;
        }
        else if (next_position.z > 14.25f)
        {
            next_position.z = 14.25f;
        }

        // move the agent to the new position
        transform.position = next_position;




    // Reward System:

        // if agent crosses Obstacle 3
        if (this.transform.localPosition.x > 26f)
        {
            // if agent reaches end_goal
            if (Vector3.Distance(this.transform.localPosition, End_Goal.localPosition) < 2.5f)
            {
                SetReward(1f);
                EndEpisode();
            }

            // if agent moves closer to end_goal
            if (Vector3.Distance(this.transform.localPosition, End_Goal.localPosition) < prev_goal_distance)
            {
                    SetReward(0.02f);
            }
            else 
            {
                SetReward(-0.02f);
            }
        }

        // if agent is between obstacle_2 and obstacle_3
        else if (this.transform.localPosition.x > 17.5f)
        {
            // if obstacle_3 is moving up
            if (obstacle_direction_3 == 1)
            {
                // if agent moves in the opposite direction to obstacle_3
                if (Vector3.Distance(this.transform.localPosition, target_down_3) < prev_down_distance_3)
                {
                    SetReward(0.02f);
                }
                else 
                {
                    SetReward(-0.02f);
                }
            }
            // if obstacle_3 is moving down
            else if (obstacle_direction_3 == -1)
            {
                // if agent moves in the opposite direction to obstacle_3
                if (Vector3.Distance(this.transform.localPosition, target_up_3) < prev_up_distance_3)
                {
                    SetReward(0.02f);
                }
                else 
                {
                    SetReward(-0.02f);
                }
            }
        }
        
        // if agent is between obstacle_1 and obstacle_2
        else if (this.transform.localPosition.x > 10f)
        {
            // if obstacle_2 is moving up
            if (obstacle_direction_2 == 1)
            {
                // if agent moves in the opposite direction to obstacle_2
                if (Vector3.Distance(this.transform.localPosition, target_down_2) < prev_down_distance_2)
                {
                    SetReward(0.02f);
                }
                else 
                {
                    SetReward(-0.02f);
                }
            }
            // if obstacle_2 is moving down
            else if (obstacle_direction_2 == -1)
            {
                // if agent moves in the opposite direction to obstacle_2
                if (Vector3.Distance(this.transform.localPosition, target_up_2) < prev_up_distance_2)
                {
                    SetReward(0.02f);
                }
                else 
                {
                    SetReward(-0.02f);
                }
            }
        }

        // if agent is before obstacle_1
        else 
        {
            // if obstacle_1 is moving up
            if (obstacle_direction_1 == 1)
            {
                // if agent moves in the opposite direction to obstacle_1
                if (Vector3.Distance(this.transform.localPosition, target_down_1) < prev_down_distance_1)
                {
                    SetReward(0.02f);
                }
                else 
                {
                    SetReward(-0.02f);
                }
            }
            // if obstacle_1 is moving down
            else if (obstacle_direction_1 == -1)
            {
                // if agent moves in the opposite direction to obstacle_1
                if (Vector3.Distance(this.transform.localPosition, target_up_1) < prev_up_distance_1)
                {
                    SetReward(0.02f);
                }
                else 
                {
                    SetReward(-0.02f);
                }
            }

        }
        
        // Calculating the current distances between the agent and the targets
        prev_up_distance_1 = Vector3.Distance(this.transform.localPosition, target_up_1);
        prev_down_distance_1 = Vector3.Distance(this.transform.localPosition, target_down_1);

        prev_up_distance_2 = Vector3.Distance(this.transform.localPosition, target_up_2);
        prev_down_distance_2 = Vector3.Distance(this.transform.localPosition, target_down_2);

        prev_up_distance_3 = Vector3.Distance(this.transform.localPosition, target_up_3);
        prev_down_distance_3 = Vector3.Distance(this.transform.localPosition, target_down_3);

        prev_goal_distance = Vector3.Distance(this.transform.localPosition, End_Goal.localPosition);
    }



    public override void Heuristic(in ActionBuffers actionsOut)
    /*
    This method sets the keyboard keys to control the agent
    */
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = k_NoAction;
        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = k_Up;
        }
        if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = k_Down;
        }
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[0] = k_Right;
        }
    }
}

