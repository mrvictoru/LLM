###################################################################################################
# Entities and Relationships Extraction Prompt
###################################################################################################

extraction_json_formatting_prompt = """
{
  "entities": [
    {
      "entity_name": "Entity1",
      "entity_type": "Type1",
      "entity_description": "Description of Entity1's attributes and activities."
    },
    {
      "entity_name": "Entity2",
      "entity_type": "Type2",
      "entity_description": "Description of Entity2's attributes and activities."
    }
    // Add more entities as needed
  ],
  "relationships": [
    {
      "source_entity": "Entity1",
      "target_entity": "Entity2",
      "relationship_description": "Explanation of why Entity1 and Entity2 are related.",
      "relationship_strength": 0.9
    },
    {
      "source_entity": "Entity2",
      "target_entity": "Entity3",
      "relationship_description": "Explanation of why Entity2 and Entity3 are related.",
      "relationship_strength": 0.7
    }
    // Add more relationships as needed
  ]
}
"""

extraction_example_1_prompt = """
TEXT: "Alice, a software engineer at TechCorp, collaborated with Bob, a data scientist at DataWorks, on a machine learning project. The project aimed to improve the recommendation system for an e-commerce platform.
"

JSON OUPUT:
{
  "entities": [
    {
      "entity_name": "Alice",
      "entity_type": "Person",
      "entity_description": "A software engineer at TechCorp."
    },
    {
      "entity_name": "TechCorp",
      "entity_type": "Organization",
      "entity_description": "A technology company where Alice works."
    },
    {
      "entity_name": "Bob",
      "entity_type": "Person",
      "entity_description": "A data scientist at DataWorks."
    },
    {
      "entity_name": "DataWorks",
      "entity_type": "Organization",
      "entity_description": "A data science company where Bob works."
    },
    {
      "entity_name": "Machine Learning Project",
      "entity_type": "Project",
      "entity_description": "A project aimed to improve the recommendation system for an e-commerce platform."
    }
  ],
  "relationships": [
    {
      "source_entity": "Alice",
      "target_entity": "TechCorp",
      "relationship_description": "Alice works at TechCorp.",
      "relationship_strength": 0.9
    },
    {
      "source_entity": "Bob",
      "target_entity": "DataWorks",
      "relationship_description": "Bob works at DataWorks.",
      "relationship_strength": 0.9
    },
    {
      "source_entity": "Alice",
      "target_entity": "Bob",
      "relationship_description": "Alice collaborated with Bob on a machine learning project.",
      "relationship_strength": 0.8
    },
    {
      "source_entity": "Machine Learning Project",
      "target_entity": "Alice",
      "relationship_description": "Alice worked on the machine learning project.",
      "relationship_strength": 0.7
    },
    {
      "source_entity": "Machine Learning Project",
      "target_entity": "Bob",
      "relationship_description": "Bob worked on the machine learning project.",
      "relationship_strength": 0.7
    }
  ]
}
"""

extraction_example_2_prompt = """
TEXT: "Dr. Smith, a renowned cardiologist at HeartCare Hospital, published a research paper with Dr. Johnson, a neurologist at BrainHealth Institute, on the effects of stress on heart health.
"

JSON OUPUT: 
{
  "entities": [
    {
      "entity_name": "Dr. Smith",
      "entity_type": "Person",
      "entity_description": "A renowned cardiologist at HeartCare Hospital."
    },
    {
      "entity_name": "HeartCare Hospital",
      "entity_type": "Organization",
      "entity_description": "A hospital specializing in heart care where Dr. Smith works."
    },
    {
      "entity_name": "Dr. Johnson",
      "entity_type": "Person",
      "entity_description": "A neurologist at BrainHealth Institute."
    },
    {
      "entity_name": "BrainHealth Institute",
      "entity_type": "Organization",
      "entity_description": "An institute specializing in brain health where Dr. Johnson works."
    },
    {
      "entity_name": "Research Paper",
      "entity_type": "Publication",
      "entity_description": "A research paper on the effects of stress on heart health."
    }
  ],
  "relationships": [
    {
      "source_entity": "Dr. Smith",
      "target_entity": "HeartCare Hospital",
      "relationship_description": "Dr. Smith works at HeartCare Hospital.",
      "relationship_strength": 0.9
    },
    {
      "source_entity": "Dr. Johnson",
      "target_entity": "BrainHealth Institute",
      "relationship_description": "Dr. Johnson works at BrainHealth Institute.",
      "relationship_strength": 0.9
    },
    {
      "source_entity": "Dr. Smith",
      "target_entity": "Dr. Johnson",
      "relationship_description": "Dr. Smith published a research paper with Dr. Johnson.",
      "relationship_strength": 0.8
    },
    {
      "source_entity": "Research Paper",
      "target_entity": "Dr. Smith",
      "relationship_description": "Dr. Smith authored the research paper.",
      "relationship_strength": 0.7
    },
    {
      "source_entity": "Research Paper",
      "target_entity": "Dr. Johnson",
      "relationship_description": "Dr. Johnson authored the research paper.",
      "relationship_strength": 0.7
    }
  ]
}
"""

graph_extraction_prompt ="""
Extract entities and relationships from the following text and format them for a knowledge graph. Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Appropriate entity type
- entity_description: Comprehensive description of the entity's attributes and activities
2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
3. Return output in list of json of all the entities and relationships identified in steps 1 and 2, if no entities and relationships is identify, return an empty json.

JSON Format:
{extraction_json_formatting_prompt}

**Example 1:**
{extraction_example_1_prompt}

**Example 2:**
{extraction_example_2_prompt}

Now, based on these examples, please extract the entities and relationships from the following text and return the information in the same structured format:

Here is the text:{text}

JSON OUPUT:
"""
###################################################################################################
# Duplicate Entities Detection Prompt
###################################################################################################

check_duplicate_entities_prompt = """
Are the following two entities duplicates?

Entity 1:
Name: {entity1_name}
Type: {entity1_type}
Description: {entity1_description}

Entity 2:
Name: {entity2_name}
Type: {entity2_type}
Description: {entity2_description}

Answer 'yes' or 'no' only.
"""

summarize_descriptions_prompt = """
Summarize the following two descriptions into one coherent description:

Description 1: {description1}

Description 2: {description2}

Only provide the summary in the response.
Summary:
"""

###################################################################################################
# Community Report Generation Prompt
###################################################################################################

community_report_generation_prompt = """
You are an AI assistant that aids a human analyst in information discovery. This involves identifying and assessing relevant information related to entities (e.g., organizations, individuals) within a network.

# Goal
Generate a comprehensive community report based on a list of entities and their relationships, including optional claims. The report will inform decision-makers about the community's key entities, legal compliance, technical capabilities, reputation, and noteworthy claims.

# Report Structure
The report should include:
- **TITLE**: A short, specific name representing key entities, including representative named entities when possible.
- **SUMMARY**: An executive summary of the community's structure, entity relationships, and significant information.
- **IMPACT SEVERITY RATING**: A float score (0-10) indicating the community's impact severity.
- **RATING EXPLANATION**: A single sentence explaining the impact severity rating.
- **DETAILED FINDINGS**: 5-10 key insights about the community, each with a summary and explanatory text following the grounding rules below.

Return the output as a well-formed JSON-formatted string using the following format:
{community_report_format_prompt}

# Grounding Rules
- Support points with data references like:
  "Example sentence supported by data [Data: <dataset name> (record ids); <dataset name> (record ids)]."
- Limit to 5 record ids per reference, adding "+more" if necessary.
- Example:
  "Person X owns Company Y [Data: Reports (1), Entities (5,7); Relationships (23); Claims (7,2,34,64,46, +more)]."
- Exclude unsupported information.

# Example Input
-----------
{community_report_example_prompt}

# Real Data
Use the following Subgraph Data for your answer. Do not fabricate information.

Subgraph Data:
{input_text}

Output:
"""

community_report_format_prompt = """{
  "title": <report_title>,
  "summary": <executive_summary>,
  "rating": <impact_severity_rating>,
  "rating_explanation": <rating_explanation>,
  "findings": [
      {
          "summary":<insight_1_summary>,
          "explanation": <insight_1_explanation>
      },
      {
          "summary":<insight_2_summary>,
          "explanation": <insight_2_explanation>
      }
  ]
}"""

community_report_example_prompt = """
First example Subgraph Data:

Entities:
id,entity,description
5,VERDANT OASIS PLAZA,Verdant Oasis Plaza is the location of the Unity March
6,HARMONY ASSEMBLY,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza

Relationships:
id,source,target,description
37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza is the location of the Unity March
38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza
39,VERDANT OASIS PLAZA,UNITY MARCH,The Unity March is taking place at Verdant Oasis Plaza
40,VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight is reporting on the Unity march taking place at Verdant Oasis Plaza
41,VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi is speaking at Verdant Oasis Plaza about the march
43,HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly is organizing the Unity March

Output:
{
  "title": "Verdant Oasis Plaza and Unity March",
  "summary": "The community revolves around the Verdant Oasis Plaza, which is the location of the Unity March. The plaza has relationships with the Harmony Assembly, Unity March, and Tribune Spotlight, all of which are associated with the march event.",
  "rating": 5.0,
  "rating_explanation": "The impact severity rating is moderate due to the potential for unrest or conflict during the Unity March.",
  "findings": [
      {
          "summary": "Verdant Oasis Plaza as the central location",
          "explanation": "Verdant Oasis Plaza is the central entity in this community, serving as the location for the Unity March. This plaza is the common link between all other entities, suggesting its significance in the community. The plaza's association with the march could potentially lead to issues such as public disorder or conflict, depending on the nature of the march and the reactions it provokes. [Data: Entities (5), Relationships (37, 38, 39, 40, 41,+more)]"
      },
      {
          "summary": "Harmony Assembly's role in the community",
          "explanation": "Harmony Assembly is another key entity in this community, being the organizer of the march at Verdant Oasis Plaza. The nature of Harmony Assembly and its march could be a potential source of threat, depending on their objectives and the reactions they provoke. The relationship between Harmony Assembly and the plaza is crucial in understanding the dynamics of this community. [Data: Entities(6), Relationships (38, 43)]"
      },
      {
          "summary": "Unity March as a significant event",
          "explanation": "The Unity March is a significant event taking place at Verdant Oasis Plaza. This event is a key factor in the community's dynamics and could be a potential source of threat, depending on the nature of the march and the reactions it provokes. The relationship between the march and the plaza is crucial in understanding the dynamics of this community. [Data: Relationships (39)]"
      },
      {
          "summary": "Role of Tribune Spotlight",
          "explanation": "Tribune Spotlight is reporting on the Unity March taking place in Verdant Oasis Plaza. This suggests that the event has attracted media attention, which could amplify its impact on the community. The role of Tribune Spotlight could be significant in shaping public perception of the event and the entities involved. [Data: Relationships (40)]"
      }
  ]
}

Second example Subgraph Data:

Entities:
id,entity,description
1,LIBERTY HALL,Liberty Hall serves as a community center for various local events
2,PEACEMAKER SOCIETY,Peacemaker Society is dedicated to promoting harmony within the community

Relationships:
id,source,target,description

Output:
{
  "title": "Liberty Hall and Peacemaker Society",
  "summary": "The community includes Liberty Hall, a central location for local events, and the Peacemaker Society, an organization dedicated to promoting harmony. There are no direct relationships between the entities.",
  "rating": 3.0,
  "rating_explanation": "The impact severity rating is low as there are no existing relationships that could lead to potential conflicts or issues.",
  "findings": [
      {
          "summary": "Liberty Hall as a community center",
          "explanation": "Liberty Hall is a key entity in the community, serving as a hub for various local events. Its role as a central location fosters community engagement and participation. [Data: Entities (1)]"
      },
      {
          "summary": "Peacemaker Society's mission",
          "explanation": "Peacemaker Society is focused on promoting harmony within the community. Its efforts contribute to maintaining a peaceful and cooperative environment. [Data: Entities (2)]"
      }
  ]
}"""