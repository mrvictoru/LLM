json_formatting_prompt = """
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

example_1_prompt = """
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

example_2_prompt = """
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
{json_formatting_prompt}

**Example 1:**
{example_1_prompt}

**Example 2:**
{example_2_prompt}

Now, based on these examples, please extract the entities and relationships from the following text and return the information in the same structured format:

Here is the text:{text}

JSON OUPUT:
"""

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