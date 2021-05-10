-- MySQL dump 10.13  Distrib 8.0.20, for Win64 (x86_64)
--
-- Host: localhost    Database: property_schema
-- ------------------------------------------------------
-- Server version	8.0.20

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `student`
--

DROP TABLE IF EXISTS `student`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `student` (
  `ID_S` varchar(20) NOT NULL,
  `Name` varchar(20) NOT NULL,
  `Degree` varchar(10) NOT NULL,
  `Salary` int NOT NULL,
  `LAB_No` varchar(10) NOT NULL,
  PRIMARY KEY (`ID_S`),
  KEY `LAB_No` (`LAB_No`),
  CONSTRAINT `student_ibfk_1` FOREIGN KEY (`LAB_No`) REFERENCES `lab` (`L_No`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `student`
--

LOCK TABLES `student` WRITE;
/*!40000 ALTER TABLE `student` DISABLE KEYS */;
INSERT INTO `student` VALUES ('N94074001','Bob','Master',3000,'L0003'),('N94084013','Jimmy','PhD',4000,'L0004'),('N94084039','Hebe','Master',3500,'L0005'),('N95062121','Ruby','Master',3000,'L0003'),('N95074026','Alex','PhD',3500,'L0007'),('N96054019','Peter','Master',3000,'L0001'),('N96084021','Alice','Master',4000,'L0002'),('N96084037','Jacky','Master',4000,'L0002'),('N96084062','Sandy','Master',4000,'L0002'),('N96084117','Mandy','Master',0,'L0006'),('N96085027','Bryan','PhD',4000,'L0010'),('N97074102','Amanda','Master',2000,'L0009'),('N97082121','Jack','Master',3500,'L0004'),('N97084026','Emily','Master',3500,'L0005'),('N97084056','Poter','PhD',4500,'L0005'),('N97085063','Jacky','PhD',4000,'L0010'),('N98054021','Cindy','PhD',3300,'L0007'),('N98064043','Ryan','Master',4000,'L0008'),('N98074021','Peter','Master',0,'L0006'),('N98075111','Linda','PhD',4000,'L0009'),('N98082033','Tiffany','Master',3000,'L0001'),('N98084094','Aron','PhD',4000,'L0003'),('N98084118','Tom','Master',3000,'L0008'),('N99054134','Claire','Master',4000,'L0002');
/*!40000 ALTER TABLE `student` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-06-11 14:33:05
